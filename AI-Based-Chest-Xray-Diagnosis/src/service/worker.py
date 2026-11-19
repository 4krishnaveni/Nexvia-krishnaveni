"""Polls Orthanc for newly stable studies, runs inference, posts results to the API.

    python -m src.service.worker

Idempotent: remembers the last /changes sequence number and the set of processed study ids
in a state file, and the API ignores duplicate study ids too.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import requests

from src.data.dicom_utils import read_dicom
from src.service.inference import Predictor, env_flag, save_outputs
from src.service.orthanc_client import OrthancClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("worker")


class State:
    def __init__(self, path: Path):
        self.path = path
        self.since, self.done, self.fails = 0, set(), {}
        if path.exists():
            d = json.loads(path.read_text())
            self.since, self.done, self.fails = d["since"], set(d["done"]), d.get("fails", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"since": self.since, "done": sorted(self.done), "fails": self.fails}))
        tmp.replace(self.path)


def process_study(study_id: str, orthanc: OrthancClient, predictor: Predictor, results_dir: Path,
                  api_url: str, api_key: str) -> None:
    instances = orthanc.study_instances(study_id)
    if not instances:
        raise RuntimeError("study has no instances")
    ds = read_dicom(orthanc.instance_file(instances[0]["ID"]))  # CXR = single frontal image
    res = predictor.predict(ds)
    files = save_outputs(res, results_dir / study_id)
    payload = {
        "study_id": study_id,
        "patient_id": str(getattr(ds, "PatientID", "")),
        "study_description": str(getattr(ds, "StudyDescription", "")),
        "probability": res["prob"], "predicted_label": res["label"], "threshold": res["threshold"],
        "model_version": res["version"], "files": files, "untrained": predictor.untrained,
        "mask_area_fraction": float(res["mask"].mean()),
    }
    r = requests.post(f"{api_url}/internal/results", json=payload, headers={"X-API-Key": api_key}, timeout=30)
    r.raise_for_status()
    log.info("study %s -> p=%.3f (%s)", study_id, res["prob"], r.json().get("status"))


MAX_FAILS = 3


def run_once(orthanc: OrthancClient, predictor: Predictor, state: State, results_dir: Path,
             api_url: str, api_key: str) -> int:
    """One polling pass. Returns number of studies processed.
    A failing study is retried on the next poll (the /changes cursor is not advanced) up to
    MAX_FAILS times, then skipped so one bad file cannot block the queue forever."""
    n = 0
    while True:
        ch = orthanc.changes(since=state.since, limit=100)
        blocked = False
        for c in ch["Changes"]:
            sid = c["ID"]
            if c["ChangeType"] != "StableStudy" or sid in state.done:
                continue
            try:
                process_study(sid, orthanc, predictor, results_dir, api_url, api_key)
                state.done.add(sid)
                n += 1
            except Exception:
                state.fails[sid] = state.fails.get(sid, 0) + 1
                log.exception("failed to process study %s (attempt %d)", sid, state.fails[sid])
                if state.fails[sid] >= MAX_FAILS:
                    log.error("giving up on study %s", sid)
                    state.done.add(sid)
                else:
                    blocked = True
        if blocked:
            state.save()
            return n
        state.since = ch["Last"]
        state.save()
        if ch["Done"]:
            return n


def main() -> None:
    orthanc = OrthancClient(os.getenv("ORTHANC_URL", "http://localhost:8042"),
                            os.getenv("ORTHANC_USER"), os.getenv("ORTHANC_PASSWORD"))
    api_url = os.getenv("API_URL", "http://localhost:8000")
    api_key = os.environ["INTERNAL_API_KEY"]
    results_dir = Path(os.getenv("RESULTS_DIR", "results"))
    predictor = Predictor(os.getenv("CLS_WEIGHTS", "models/classifier.pt"),
                          os.getenv("SEG_WEIGHTS", "models/unet.pt"),
                          allow_untrained=env_flag("ALLOW_UNTRAINED"))
    state = State(Path(os.getenv("STATE_FILE", "state/worker_state.json")))
    poll = float(os.getenv("POLL_SECONDS", "5"))
    log.info("worker started (model %s)", predictor.version)
    while True:
        try:
            run_once(orthanc, predictor, state, results_dir, api_url, api_key)
        except requests.RequestException as e:
            log.warning("service not reachable yet: %s", e)
        time.sleep(poll)


if __name__ == "__main__":
    main()
