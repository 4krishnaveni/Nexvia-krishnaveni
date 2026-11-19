import os
import sqlite3
import tempfile

import pytest

_tmp = tempfile.mkdtemp()
os.environ.update(DB_PATH=f"{_tmp}/t.db", RESULTS_DIR=f"{_tmp}/results", JWT_SECRET="x" * 32,
                  INTERNAL_API_KEY="k", RADIOLOGIST_USER="rad", RADIOLOGIST_PASSWORD="pw")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402

RESULT = {"study_id": "abc123", "probability": 0.55, "predicted_label": 1, "threshold": 0.5,
          "model_version": "m1", "files": {"image": "image.png", "heatmap": "heatmap.png", "mask": "mask.png"}}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth(client):
    r = client.post("/auth/login", json={"username": "rad", "password": "pw"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_login_rejects_wrong_password(client):
    assert client.post("/auth/login", json={"username": "rad", "password": "nope"}).status_code == 401


def test_studies_require_auth(client):
    assert client.get("/studies").status_code in (401, 403)


def test_internal_endpoint_needs_key_and_is_idempotent(client, auth):
    assert client.post("/internal/results", json=RESULT).status_code == 401
    assert client.post("/internal/results", json=RESULT, headers={"X-API-Key": "k"}).json()["status"] == "stored"
    assert client.post("/internal/results", json=RESULT, headers={"X-API-Key": "k"}).json()["status"] == "duplicate_ignored"
    s = client.get("/studies/abc123", headers=auth).json()
    assert s["uncertain"] is True and s["status"] == "PENDING_REVIEW"


def test_decision_writes_audit_row_and_audit_is_append_only(client, auth):
    r = client.post("/studies/abc123/decision", json={"decision": "approve", "comment": "ok"}, headers=auth)
    assert r.json()["status"] == "APPROVED"
    rows = client.get("/audit", headers=auth).json()
    assert rows[0]["user"] == "rad" and rows[0]["confidence_at_time"] == 0.55 and rows[0]["model_version"] == "m1"
    con = sqlite3.connect(os.environ["DB_PATH"])
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("UPDATE audit_log SET decision='reject'")
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("DELETE FROM audit_log")


def test_invalid_decision_rejected(client, auth):
    assert client.post("/studies/abc123/decision", json={"decision": "maybe"}, headers=auth).status_code == 422
