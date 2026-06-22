"""FastAPI inference layer for PoissonGuard.

Loads a trained detector (path from the ``POISSONGUARD_BASELINES`` env var, or
``artifacts/baselines.json``) and exposes:

* ``GET  /health``           -- liveness + number of loaded baselines
* ``POST /score``            -- score a window of events
* ``GET  /baselines/{key}``  -- inspect a learned baseline

Run:
    uvicorn poissonguard.serving:app --reload
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .detector import Detector

DEFAULT_BASELINES = os.environ.get("POISSONGUARD_BASELINES", "artifacts/baselines.json")

app = FastAPI(title="PoissonGuard", version="0.1.0",
              description="Poisson-process anomaly detection hardened against baseline poisoning.")

from .dashboard import router as dashboard_router  # noqa: E402
app.include_router(dashboard_router)


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    if not os.path.exists(DEFAULT_BASELINES):
        raise FileNotFoundError(
            f"baselines not found at {DEFAULT_BASELINES!r}; run scripts/train.py first "
            f"or set POISSONGUARD_BASELINES."
        )
    return Detector.load(DEFAULT_BASELINES)


class ScoreRequest(BaseModel):
    entity: str = Field(..., examples=["entity_00"])
    event_type: str = Field(..., examples=["login"])
    timestamps: list[float] = Field(..., description="Event epoch seconds.")
    start: float = Field(..., description="Observation window start (epoch seconds).")
    end: float = Field(..., description="Observation window end (epoch seconds).")
    update_baseline: bool = Field(False, description="Let the drift gate fold this window in if benign.")


class SubScoreOut(BaseModel):
    name: str
    p_value: float
    statistic: float
    detail: str


class BucketResult(BaseModel):
    bucket: str
    observed_count: int
    expected_count: float
    fused_p_value: float
    severity: str
    drift_decision: str
    is_anomaly: bool
    sub_scores: list[SubScoreOut]
    detail: str


class ScoreResponse(BaseModel):
    entity: str
    event_type: str
    is_anomaly: bool
    results: list[BucketResult]


@app.get("/health")
def health() -> dict:
    try:
        det = get_detector()
        return {"status": "ok", "baselines": len(det.baselines)}
    except FileNotFoundError as e:
        return {"status": "degraded", "detail": str(e)}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    try:
        det = get_detector()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    results = det.detect(
        req.entity, req.event_type, req.timestamps, req.start, req.end,
        update_baseline=req.update_baseline,
    )
    bucket_results = [
        BucketResult(
            bucket=r.bucket.value,
            observed_count=r.observed_count,
            expected_count=r.expected_count,
            fused_p_value=r.fused_p_value,
            severity=r.severity.value,
            drift_decision=r.drift_decision.value,
            is_anomaly=r.is_anomaly,
            sub_scores=[SubScoreOut(name=s.name, p_value=s.p_value,
                                    statistic=s.statistic, detail=s.detail)
                        for s in r.sub_scores],
            detail=r.detail,
        )
        for r in results
    ]
    return ScoreResponse(
        entity=req.entity,
        event_type=req.event_type,
        is_anomaly=any(b.is_anomaly for b in bucket_results),
        results=bucket_results,
    )


@app.get("/baselines/{key}")
def get_baseline(key: str) -> dict:
    det = get_detector()
    baseline = det.baselines.get(key)
    if baseline is None:
        raise HTTPException(status_code=404, detail=f"no baseline for key {key!r}")
    out = baseline.to_dict()
    guard = det.guards.get(key)
    if guard is not None:
        out["guard"] = guard.state_dict()
    return out
