import hashlib
import hmac
import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.getenv("DB_PATH", "/data/app.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS studies (
  study_id TEXT PRIMARY KEY, patient_id TEXT, study_description TEXT,
  probability REAL NOT NULL, predicted_label INTEGER NOT NULL, threshold REAL NOT NULL,
  model_version TEXT NOT NULL, files TEXT NOT NULL, untrained INTEGER DEFAULT 0,
  mask_area_fraction REAL, status TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, study_id TEXT NOT NULL, user TEXT NOT NULL,
  decision TEXT NOT NULL, comment TEXT, model_version TEXT NOT NULL,
  confidence_at_time REAL NOT NULL,
  timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
-- append-only: the database itself refuses edits and deletes
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""


def hash_password(pw: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)
    return salt.hex() + "$" + dk.hex()


def verify_password(pw: str, stored: str) -> bool:
    salt_hex, _ = stored.split("$")
    return hmac.compare_digest(hash_password(pw, bytes.fromhex(salt_hex)), stored)


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db(username: str, password: str) -> None:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with conn() as c:
        c.executescript(SCHEMA)
        c.execute(
            "INSERT OR REPLACE INTO users(username, password_hash, role) VALUES (?,?,?)",
            (username, hash_password(password), "radiologist"),
        )
