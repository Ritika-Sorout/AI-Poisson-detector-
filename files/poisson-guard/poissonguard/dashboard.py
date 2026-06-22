"""FastAPI-served dashboard for PoissonGuard.

Adds a router with a small JSON API and a single embedded HTML page (Chart.js
via CDN -- no Node build step). Mounted by ``serving.py`` so the whole thing
runs with the same ``uvicorn`` command.

Endpoints
---------
* ``GET /``               -- the dashboard page
* ``GET /api/summary``    -- benchmark AUCs, per-attack rates, poisoning summary
* ``GET /api/poisoning``  -- per-day boil-the-frog trace
* ``GET /api/baselines``  -- learned baseline table
* ``GET /api/scenario``   -- generate + score a window (live demo)
"""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from .bucketing import BucketingConfig
from .evaluation import build_trained_pair, evaluate_detection, evaluate_poisoning, poisoning_trace
from .generators import (
    ATTACK_TYPES,
    DAY,
    _business_day_starts,
    generate_eval_windows,
    generate_normal,
    make_attack,
)
from .schemas import Bucket

router = APIRouter()
_CFG = BucketingConfig()
_SCENARIOS = ("normal",) + ATTACK_TYPES


@lru_cache(maxsize=1)
def _ctx() -> SimpleNamespace:
    """Train once, compute benchmark + poisoning trace. Cached for the process."""
    detector, legacy, population = build_trained_pair(weeks=4, n_entities=12, seed=0, bucketing=_CFG)
    windows = generate_eval_windows(population, _CFG, seed=100)
    report = evaluate_detection(detector, legacy, windows)

    # Poisoning runs on a *separate* trained pair so the scenario/baseline
    # detector stays pristine (un-poisoned) for the live demo.
    pdet, pleg, _ = build_trained_pair(weeks=4, n_entities=12, seed=0, bucketing=_CFG)
    pr = evaluate_poisoning(pdet, pleg, population[0], days=40, end_mult=3.0)
    pdet2, pleg2, _ = build_trained_pair(weeks=4, n_entities=12, seed=0, bucketing=_CFG)
    trace = poisoning_trace(pdet2, pleg2, population[0], days=40, end_mult=3.0)

    return SimpleNamespace(detector=detector, population=population, report=report,
                           poisoning=pr, trace=trace, demo_base=_business_day_starts(300 * 7 * DAY, _CFG, 1)[0])


@router.get("/api/summary")
def api_summary() -> dict:
    ctx = _ctx()
    r = ctx.report
    pr = ctx.poisoning
    p0 = ctx.population[0]
    return {
        "auc": {"full": r.auc_full, "rate_only": r.auc_rate_only, "legacy": r.auc_legacy},
        "ap": {"full": r.ap_full, "legacy": r.ap_legacy},
        "per_attack": {a: {"full": r.per_attack_full[a], "legacy": r.per_attack_legacy[a]}
                       for a in ATTACK_TYPES},
        "poisoning": {
            "entity": p0.entity, "baseline_rate": round(p0.business_rate, 2),
            "target_rate": round(pr.target_rate, 2), "freeze_day": pr.freeze_day,
            "guard_anchor": round(pr.guard_anchor, 2), "guard_frozen": pr.guard_frozen,
            "legacy_lambda_initial": round(pr.legacy_lambda_initial, 2),
            "legacy_lambda_final": round(pr.legacy_lambda_final, 2),
            "full_detects_target": pr.full_detects_target,
        },
        "entities": [p.entity for p in ctx.population],
        "scenarios": list(_SCENARIOS),
    }


@router.get("/api/poisoning")
def api_poisoning() -> dict:
    return {"trace": _ctx().trace}


@router.get("/api/baselines")
def api_baselines() -> dict:
    ctx = _ctx()
    rows = []
    for key, b in sorted(ctx.detector.baselines.items()):
        guard = ctx.detector.guards.get(key)
        rows.append({
            "key": key, "entity": b.entity, "event_type": b.event_type,
            "bucket": b.bucket.value, "rate": round(b.posterior_mean_rate, 3),
            "n_events": b.n_events, "exposure_hours": round(b.exposure_hours, 1),
            "frozen": bool(guard.frozen) if guard else False,
        })
    return {"baselines": rows}


@router.get("/api/scenario")
def api_scenario(
    type: str = Query("volume_spike"),
    entity: str = Query(""),
    severity: float = Query(6.0),
) -> dict:
    ctx = _ctx()
    profile = next((p for p in ctx.population if p.entity == entity), ctx.population[0])
    base = ctx.demo_base
    rng = np.random.default_rng(7)

    if type == "normal":
        ts = generate_normal(profile, base, base + DAY, _CFG, rng)
    else:
        ts = make_attack(type, profile, base, _CFG, rng, severity=severity)

    results = ctx.detector.detect(profile.entity, profile.event_type, ts, base, base + DAY)

    # hourly histogram (hour-of-day 0..23)
    hours = (((np.asarray(ts) - base) / 3600.0) % 24).astype(int) if len(ts) else np.empty(0)
    hist = [int((hours == h).sum()) for h in range(24)]

    return {
        "entity": profile.entity, "event_type": profile.event_type, "type": type,
        "total_events": int(len(ts)), "hourly": hist,
        "results": [{
            "bucket": r.bucket.value, "observed_count": r.observed_count,
            "expected_count": round(r.expected_count, 2), "fused_p_value": r.fused_p_value,
            "severity": r.severity.value, "drift_decision": r.drift_decision.value,
            "is_anomaly": r.is_anomaly,
            "sub_scores": [{"name": s.name, "p_value": s.p_value,
                            "statistic": round(s.statistic, 3) if s.statistic == s.statistic else None,
                            "detail": s.detail} for s in r.sub_scores],
        } for r in results],
    }


@router.get("/", response_class=HTMLResponse)
def dashboard_page() -> str:
    return _HTML


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PoissonGuard Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0b1020;--panel:#141a2e;--panel2:#1b2440;--ink:#e7ecf5;--muted:#8b97b5;
--accent:#5b8cff;--good:#28c391;--warn:#ffb454;--bad:#ff5d6c;--line:#2a3454;}
*{box-sizing:border-box}
body{margin:0;background:linear-gradient(180deg,#0b1020,#0a0e1c);color:var(--ink);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
header{padding:28px 32px 12px;border-bottom:1px solid var(--line);}
h1{margin:0;font-size:22px;letter-spacing:.3px}
.sub{color:var(--muted);margin-top:4px;max-width:820px}
.wrap{padding:24px 32px;display:grid;gap:20px;grid-template-columns:repeat(12,1fr)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 18px 16px;}
.card h2{margin:0 0 12px;font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.col-12{grid-column:span 12}.col-8{grid-column:span 8}.col-6{grid-column:span 6}.col-4{grid-column:span 4}
@media(max-width:980px){.col-8,.col-6,.col-4{grid-column:span 12}}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.kpi{background:var(--panel2);border-radius:12px;padding:14px}
.kpi .v{font-size:26px;font-weight:700}
.kpi .l{color:var(--muted);font-size:12px;margin-top:2px}
.kpi .v.good{color:var(--good)}.kpi .v.bad{color:var(--bad)}.kpi .v.warn{color:var(--warn)}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
select,button{background:var(--panel2);color:var(--ink);border:1px solid var(--line);
border-radius:9px;padding:8px 12px;font-size:13px}
button{background:var(--accent);border:none;cursor:pointer;font-weight:600}
button:hover{filter:brightness(1.08)}
input[type=range]{accent-color:var(--accent)}
.verdicts{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.verdict{flex:1;min-width:220px;background:var(--panel2);border-radius:12px;padding:14px;border-left:4px solid var(--line)}
.verdict.anom{border-left-color:var(--bad)}.verdict.ok{border-left-color:var(--good)}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase}
.badge.critical,.badge.high{background:rgba(255,93,108,.18);color:var(--bad)}
.badge.medium{background:rgba(255,180,84,.18);color:var(--warn)}
.badge.low,.badge.none{background:rgba(40,195,145,.16);color:var(--good)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:12px}
td.mono,th.mono{font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.frozen{color:var(--bad);font-weight:700}
.scroll{max-height:340px;overflow:auto}
.foot{color:var(--muted);font-size:12px;padding:0 32px 28px}
.pill{font-size:11px;color:var(--muted)}
</style>
</head>
<body>
<header>
  <h1>PoissonGuard &middot; anomaly detection hardened against baseline poisoning</h1>
  <div class="sub">Per-entity event rates modeled as Poisson processes (Gamma-Poisson Bayesian core +
  process-shape tests), with a drift-integrity gate that refuses to absorb slow "boil-the-frog"
  poisoning into its own baseline.</div>
</header>

<div class="wrap">
  <div class="card col-12">
    <div class="kpis" id="kpis"></div>
  </div>

  <div class="card col-6">
    <h2>Attack benchmark &mdash; detection rate @ 5% FPR (full vs legacy)</h2>
    <canvas id="benchChart" height="200"></canvas>
  </div>

  <div class="card col-6">
    <h2>Poisoning defense &mdash; boil-the-frog ramp</h2>
    <canvas id="poisonChart" height="200"></canvas>
  </div>

  <div class="card col-12">
    <h2>Live scoring</h2>
    <div class="controls">
      <label class="pill">Entity</label>
      <select id="entity"></select>
      <label class="pill">Scenario</label>
      <select id="scenario"></select>
      <label class="pill">Severity</label>
      <input type="range" id="sev" min="1" max="10" step="0.5" value="6"/>
      <span id="sevv" class="pill">6.0&times;</span>
      <button id="run">Score window</button>
    </div>
    <div class="verdicts" id="verdicts"></div>
    <div class="col-12" style="display:grid;grid-template-columns:1fr 1fr;gap:18px">
      <div><canvas id="hourChart" height="170"></canvas></div>
      <div class="scroll"><table id="subs"><thead><tr><th>bucket</th><th>test</th>
        <th class="mono">p-value</th><th>detail</th></tr></thead><tbody></tbody></table></div>
    </div>
  </div>

  <div class="card col-12">
    <h2>Learned baselines</h2>
    <div class="scroll"><table id="baselines"><thead><tr>
      <th>entity</th><th>event type</th><th>bucket</th><th class="mono">rate /hr</th>
      <th class="mono">n events</th><th class="mono">exposure h</th><th>gate</th>
    </tr></thead><tbody></tbody></table></div>
  </div>
</div>
<div class="foot">FastAPI-served &middot; charts via Chart.js &middot; data computed live from a freshly trained model.</div>

<script>
const fmtP = p => p < 1e-3 ? p.toExponential(2) : p.toFixed(3);
const $ = s => document.querySelector(s);

async function jget(u){const r=await fetch(u);return r.json();}

function kpi(v,l,cls){return `<div class="kpi"><div class="v ${cls||''}">${v}</div><div class="l">${l}</div></div>`;}

let benchChart, poisonChart, hourChart;

async function loadSummary(){
  const s = await jget('/api/summary');
  const p = s.poisoning;
  $('#kpis').innerHTML =
    kpi(s.auc.full.toFixed(3),'ROC-AUC (full)','good')+
    kpi(s.auc.legacy.toFixed(3),'ROC-AUC (legacy)','warn')+
    kpi('day '+p.freeze_day,'gate froze poisoning at', p.guard_frozen?'good':'bad')+
    kpi(p.guard_anchor+' / '+p.target_rate,'anchor held vs attacker target','good');

  // entity + scenario selects
  $('#entity').innerHTML = s.entities.map(e=>`<option>${e}</option>`).join('');
  $('#scenario').innerHTML = s.scenarios.map(e=>`<option>${e}</option>`).join('');
  $('#scenario').value='volume_spike';

  // benchmark chart
  const labels = Object.keys(s.per_attack);
  benchChart = new Chart($('#benchChart'),{type:'bar',
    data:{labels,datasets:[
      {label:'full',data:labels.map(a=>s.per_attack[a].full*100),backgroundColor:'#5b8cff'},
      {label:'legacy',data:labels.map(a=>s.per_attack[a].legacy*100),backgroundColor:'#ffb454'}]},
    options:{responsive:true,scales:{y:{max:100,ticks:{color:'#8b97b5',callback:v=>v+'%'},grid:{color:'#2a3454'}},
      x:{ticks:{color:'#8b97b5'},grid:{display:false}}},plugins:{legend:{labels:{color:'#e7ecf5'}}}}});
}

async function loadPoisoning(){
  const {trace} = await jget('/api/poisoning');
  const labels = trace.map(t=>t.day);
  const freeze = trace.findIndex(t=>t.frozen);
  poisonChart = new Chart($('#poisonChart'),{type:'line',
    data:{labels,datasets:[
      {label:'attacker target rate',data:trace.map(t=>t.attacker_rate),borderColor:'#ff5d6c',borderDash:[6,4],pointRadius:0,tension:.2},
      {label:'observed rate',data:trace.map(t=>t.observed_rate),borderColor:'#ffb454',pointRadius:0,tension:.2},
      {label:'gate anchor (trusted baseline)',data:trace.map(t=>t.anchor),borderColor:'#28c391',borderWidth:2,pointRadius:0,tension:.1}]},
    options:{responsive:true,
      plugins:{legend:{labels:{color:'#e7ecf5',boxWidth:12}},
        tooltip:{callbacks:{afterBody:i=>{const t=trace[i[0].dataIndex];return 'decision: '+t.decision+(t.frozen?' (FROZEN)':'');}}}},
      scales:{y:{title:{display:true,text:'events / hour',color:'#8b97b5'},ticks:{color:'#8b97b5'},grid:{color:'#2a3454'}},
        x:{title:{display:true,text:'day (freeze at day '+freeze+')',color:'#8b97b5'},ticks:{color:'#8b97b5',maxTicksLimit:12},grid:{display:false}}}}});
}

async function loadBaselines(){
  const {baselines} = await jget('/api/baselines');
  $('#baselines').querySelector('tbody').innerHTML = baselines.map(b=>
    `<tr><td>${b.entity}</td><td>${b.event_type}</td><td>${b.bucket}</td>
     <td class="mono">${b.rate}</td><td class="mono">${b.n_events}</td>
     <td class="mono">${b.exposure_hours}</td>
     <td>${b.frozen?'<span class="frozen">FROZEN</span>':'live'}</td></tr>`).join('');
}

async function runScenario(){
  const type=$('#scenario').value, entity=$('#entity').value, sev=$('#sev').value;
  const d = await jget(`/api/scenario?type=${type}&entity=${entity}&severity=${sev}`);
  $('#verdicts').innerHTML = d.results.map(r=>{
    const cls=r.is_anomaly?'anom':'ok';
    return `<div class="verdict ${cls}">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b>${r.bucket}</b><span class="badge ${r.severity}">${r.severity}</span></div>
      <div class="pill" style="margin-top:6px">observed <b>${r.observed_count}</b> &middot; expected ${r.expected_count}</div>
      <div class="pill">fused p = <b class="mono">${fmtP(r.fused_p_value)}</b></div>
      <div class="pill">gate: ${r.drift_decision} &middot; ${r.is_anomaly?'ANOMALY':'normal'}</div>
    </div>`;}).join('');

  const rows=[];
  d.results.forEach(r=>r.sub_scores.forEach(s=>rows.push(
    `<tr><td>${r.bucket}</td><td>${s.name}</td><td class="mono">${fmtP(s.p_value)}</td><td class="pill">${s.detail}</td></tr>`)));
  $('#subs').querySelector('tbody').innerHTML = rows.join('');

  const labels=[...Array(24).keys()];
  if(hourChart) hourChart.destroy();
  hourChart=new Chart($('#hourChart'),{type:'bar',
    data:{labels,datasets:[{label:'events (total '+d.total_events+')',data:d.hourly,
      backgroundColor:labels.map(h=>h>=9&&h<17?'#5b8cff':'#39456a')}]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#e7ecf5'}}},
      scales:{y:{ticks:{color:'#8b97b5'},grid:{color:'#2a3454'}},
        x:{title:{display:true,text:'hour of day (blue = business hours)',color:'#8b97b5'},
           ticks:{color:'#8b97b5',maxTicksLimit:12},grid:{display:false}}}}});
}

$('#sev').addEventListener('input',e=>$('#sevv').textContent=(+e.target.value).toFixed(1)+'\u00d7');
$('#run').addEventListener('click',runScenario);

(async()=>{await loadSummary();await loadPoisoning();await loadBaselines();await runScenario();})();
</script>
</body>
</html>
"""
