from __future__ import annotations

import requests


class OrthancClient:
    def __init__(self, url: str, user: str | None = None, password: str | None = None, timeout: int = 30):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        if user:
            self.s.auth = (user, password or "")
        self.timeout = timeout

    def _get(self, path: str, **kw):
        r = self.s.get(f"{self.url}{path}", timeout=self.timeout, **kw)
        r.raise_for_status()
        return r

    def changes(self, since: int = 0, limit: int = 100) -> dict:
        """{'Changes': [...], 'Last': int, 'Done': bool}"""
        return self._get("/changes", params={"since": since, "limit": limit}).json()

    def study(self, study_id: str) -> dict:
        return self._get(f"/studies/{study_id}").json()

    def study_instances(self, study_id: str) -> list[dict]:
        return self._get(f"/studies/{study_id}/instances").json()

    def instance_file(self, instance_id: str) -> bytes:
        return self._get(f"/instances/{instance_id}/file").content

    def upload(self, dicom_bytes: bytes) -> dict:
        r = self.s.post(f"{self.url}/instances", data=dicom_bytes,
                        headers={"Content-Type": "application/dicom"}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def alive(self) -> bool:
        try:
            self._get("/system")
            return True
        except requests.RequestException:
            return False
