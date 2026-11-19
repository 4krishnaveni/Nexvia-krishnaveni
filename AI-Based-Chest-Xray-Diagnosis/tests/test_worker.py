from pathlib import Path

from src.service import worker


class FakeOrthanc:
    def __init__(self, ids):
        self.ids = ids

    def changes(self, since=0, limit=100):
        ch = [{"ChangeType": "StableStudy", "ID": i, "Seq": n + 1} for n, i in enumerate(self.ids) if n + 1 > since]
        return {"Changes": ch, "Last": len(self.ids), "Done": True}


def test_idempotent_and_state_persists(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "process_study", lambda sid, *a, **k: calls.append(sid))
    st = worker.State(tmp_path / "s.json")
    orth = FakeOrthanc(["s1", "s2"])
    assert worker.run_once(orth, None, st, Path("."), "u", "k") == 2
    assert worker.run_once(orth, None, st, Path("."), "u", "k") == 0
    st2 = worker.State(tmp_path / "s.json")  # simulate restart
    assert worker.run_once(orth, None, st2, Path("."), "u", "k") == 0
    assert calls == ["s1", "s2"]


def test_failed_study_is_retried_then_abandoned(tmp_path, monkeypatch):
    attempts = {"n": 0}

    def flaky(sid, *a, **k):
        attempts["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(worker, "process_study", flaky)
    st = worker.State(tmp_path / "s.json")
    orth = FakeOrthanc(["bad"])
    for _ in range(5):
        worker.run_once(orth, None, st, Path("."), "u", "k")
    assert attempts["n"] == worker.MAX_FAILS and "bad" in st.done
