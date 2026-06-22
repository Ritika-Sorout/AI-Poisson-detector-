import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from poissonguard import serving

client = TestClient(serving.app)


def test_dashboard_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "PoissonGuard" in r.text
    assert "benchChart" in r.text


def test_api_summary():
    r = client.get("/api/summary")
    assert r.status_code == 200
    s = r.json()
    assert s["auc"]["full"] >= s["auc"]["legacy"]
    assert s["poisoning"]["guard_frozen"] is True
    assert s["poisoning"]["guard_anchor"] < s["poisoning"]["target_rate"]
    assert len(s["entities"]) == 12


def test_api_poisoning_trace():
    r = client.get("/api/poisoning")
    trace = r.json()["trace"]
    assert len(trace) == 40
    assert any(t["frozen"] for t in trace)
    # anchor never reaches the attacker's escalating rate
    assert trace[-1]["anchor"] < trace[-1]["attacker_rate"]


def test_api_baselines():
    r = client.get("/api/baselines")
    rows = r.json()["baselines"]
    assert len(rows) == 24
    assert {row["bucket"] for row in rows} == {"business", "offhours"}


def test_api_scenario_spike_flags_anomaly():
    r = client.get("/api/scenario", params={"type": "volume_spike", "severity": 6.0})
    d = r.json()
    assert len(d["hourly"]) == 24
    biz = [x for x in d["results"] if x["bucket"] == "business"][0]
    assert biz["is_anomaly"] is True


def test_api_scenario_normal_ok():
    r = client.get("/api/scenario", params={"type": "normal"})
    d = r.json()
    biz = [x for x in d["results"] if x["bucket"] == "business"]
    assert biz and biz[0]["is_anomaly"] is False
