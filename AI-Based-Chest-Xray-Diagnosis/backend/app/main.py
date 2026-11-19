"""FastAPI backend: study queue, radiologist decisions, append-only audit log.
Run locally:  DB_PATH=./app.db RESULTS_DIR=./results JWT_SECRET=x INTERNAL_API_KEY=y \
              RADIOLOGIST_USER=radiologist RADIOLOGIST_PASSWORD=pw uvicorn backend.app.main:app --reload
Interactive docs: http://localhost:8000/docs
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db

JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-change-me")
INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", "dev-internal-key")
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/results"))
DISCLAIMER = "FOR EDUCATION ONLY. NOT FOR CLINICAL USE."


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db(os.getenv("RADIOLOGIST_USER", "radiologist"), os.getenv("RADIOLOGIST_PASSWORD", "radiologist"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="CXR Review API", description=DISCLAIMER, version="1.0.0", lifespan=lifespan)
# Overlays are de-identified demo images, so they are served without auth so <img> tags work.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(RESULTS_DIR)), name="files")
bearer = HTTPBearer()


# ---------- models ----------
class Login(BaseModel):
    username: str
    password: str


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    comment: str = Field("", max_length=2000)


class ResultIn(BaseModel):
    study_id: str
    patient_id: str = ""
    study_description: str = ""
    probability: float = Field(ge=0, le=1)
    predicted_label: int
    threshold: float
    model_version: str
    files: dict
    untrained: bool = False
    mask_area_fraction: float | None = None


# ---------- auth ----------
def current_user(cred: HTTPAuthorizationCredentials = Depends(bearer)) -> str:
    try:
        payload = jwt.decode(cred.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "invalid or expired token")
    if payload.get("role") != "radiologist":
        raise HTTPException(403, "radiologist role required")
    return payload["sub"]


def internal_only(x_api_key: str = Header("")) -> None:
    if x_api_key != INTERNAL_KEY:
        raise HTTPException(401, "bad internal api key")


@app.post("/auth/login")
def login(body: Login):
    with db.conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (body.username,)).fetchone()
    if not row or not db.verify_password(body.password, row["password_hash"]):
        raise HTTPException(401, "wrong username or password")
    token = jwt.encode({"sub": row["username"], "role": row["role"], "exp": int(time.time()) + 8 * 3600},
                       JWT_SECRET, algorithm="HS256")
    return {"access_token": token, "token_type": "bearer", "disclaimer": DISCLAIMER}


# ---------- helpers ----------
def study_out(r) -> dict:
    files = json.loads(r["files"])
    base = f"/files/{r['study_id']}"
    return {
        "study_id": r["study_id"], "patient_id": r["patient_id"], "description": r["study_description"],
        "probability": r["probability"], "predicted_label": r["predicted_label"], "threshold": r["threshold"],
        "uncertain": 0.4 <= r["probability"] <= 0.6, "model_version": r["model_version"],
        "untrained": bool(r["untrained"]), "mask_area_fraction": r["mask_area_fraction"],
        "status": r["status"], "created_at": r["created_at"],
        "urls": {k: f"{base}/{v}" for k, v in files.items()}, "disclaimer": DISCLAIMER,
    }


# ---------- worker -> API ----------
@app.post("/internal/results", dependencies=[Depends(internal_only)])
def post_result(r: ResultIn):
    with db.conn() as c:
        if c.execute("SELECT 1 FROM studies WHERE study_id=?", (r.study_id,)).fetchone():
            return {"status": "duplicate_ignored"}  # idempotent: safe if the worker retries
        c.execute(
            "INSERT INTO studies(study_id,patient_id,study_description,probability,predicted_label,threshold,"
            "model_version,files,untrained,mask_area_fraction) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (r.study_id, r.patient_id, r.study_description, r.probability, r.predicted_label, r.threshold,
             r.model_version, json.dumps(r.files), int(r.untrained), r.mask_area_fraction),
        )
    return {"status": "stored"}


# ---------- viewer -> API ----------
@app.get("/studies")
def list_studies(status: str | None = None, user: str = Depends(current_user)):
    q, args = "SELECT * FROM studies", ()
    if status:
        q, args = q + " WHERE status=?", (status,)
    with db.conn() as c:
        rows = c.execute(q + " ORDER BY created_at DESC", args).fetchall()
    return [study_out(r) for r in rows]


@app.get("/studies/{study_id}")
def get_study(study_id: str, user: str = Depends(current_user)):
    with db.conn() as c:
        r = c.execute("SELECT * FROM studies WHERE study_id=?", (study_id,)).fetchone()
    if not r:
        raise HTTPException(404, "study not found")
    return study_out(r)


@app.post("/studies/{study_id}/decision")
def decide(study_id: str, body: Decision, user: str = Depends(current_user)):
    with db.conn() as c:
        r = c.execute("SELECT * FROM studies WHERE study_id=?", (study_id,)).fetchone()
        if not r:
            raise HTTPException(404, "study not found")
        # confidence/model are captured server-side at decision time, never trusted from the client
        c.execute(
            "INSERT INTO audit_log(study_id,user,decision,comment,model_version,confidence_at_time) VALUES (?,?,?,?,?,?)",
            (study_id, user, body.decision, body.comment, r["model_version"], r["probability"]),
        )
        status = "APPROVED" if body.decision == "approve" else "REJECTED"
        c.execute("UPDATE studies SET status=? WHERE study_id=?", (status, study_id))
    return {"study_id": study_id, "status": status}


@app.get("/audit")
def audit(study_id: str | None = None, user: str = Depends(current_user)):
    q, args = "SELECT * FROM audit_log", ()
    if study_id:
        q, args = q + " WHERE study_id=?", (study_id,)
    with db.conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY id DESC", args).fetchall()]


@app.get("/health")
def health():
    return {"ok": True}
