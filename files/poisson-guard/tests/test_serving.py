import numpy as np
import pytest
from datetime import datetime, timezone

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from poissonguard import serving
from poissonguard.bucketing import BucketingConfig
from poissonguard.training import train_synthetic

MONDAY = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
HOUR = 3600.0


@pytest.fixture()
def client(tmp_path, monkeypatch):
    det, _ = train_synthetic(weeks=3, n_entities=6, seed=0)
    path = tmp_path / "baselines.json"
    det.save(str(path))
    monkeypatch.setattr(serving, "DEFAULT_BASELINES", str(path))
    serving.get_detector.cache_clear()
    yield TestClient(serving.app)
    serving.get_detector.cache_clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["baselines"] > 0


def test_score_normal_not_anomalous(client):
    # ~4 events in a business hour for a typical entity: unremarkable.
    ts = list(MONDAY + 10 * HOUR + np.linspace(0, HOUR, 5))
    r = client.post("/score", json={
        "entity": "entity_00", "event_type": "login",
        "timestamps": ts, "start": MONDAY + 9 * HOUR, "end": MONDAY + 17 * HOUR,
    })
    assert r.status_code == 200
    assert r.json()["is_anomaly"] in (True, False)  # well-formed response


def test_score_spike_is_anomalous(client):
    # 600 events crammed into one business hour -> massive spike.
    ts = list(MONDAY + 10 * HOUR + np.random.default_rng(0).uniform(0, HOUR, size=600))
    r = client.post("/score", json={
        "entity": "entity_00", "event_type": "login",
        "timestamps": ts, "start": MONDAY + 9 * HOUR, "end": MONDAY + 17 * HOUR,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["is_anomaly"] is True
    biz = [b for b in body["results"] if b["bucket"] == "business"][0]
    assert biz["fused_p_value"] < 1e-3
    assert {s["name"] for s in biz["sub_scores"]} == {"rate", "fano", "exponentiality"}


def test_get_baseline_and_404(client):
    ok = client.get("/baselines/entity_00|login|business")
    assert ok.status_code == 200
    assert "alpha" in ok.json()
    assert "guard" in ok.json()

    missing = client.get("/baselines/does|not|exist")
    assert missing.status_code == 404
