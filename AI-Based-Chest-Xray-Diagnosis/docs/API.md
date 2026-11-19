# API summary
Live, interactive docs (OpenAPI) are served at `http://localhost:8000/docs`; export the schema with
`curl http://localhost:8000/openapi.json > docs/openapi.json`.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/auth/login` | none | `{username, password}` -> JWT (8 h) |
| GET | `/studies?status=` | JWT | Queue, newest first |
| GET | `/studies/{id}` | JWT | One study: probability, threshold, `uncertain`, overlay URLs |
| POST | `/studies/{id}/decision` | JWT | `{decision: approve\|reject, comment}`; appends to audit log |
| GET | `/audit?study_id=` | JWT | Read-only audit log |
| POST | `/internal/results` | `X-API-Key` | Worker posts a prediction (idempotent) |
| GET | `/files/{study}/{image\|heatmap\|mask}.png` | none (de-identified demo data) | Images for the viewer |

Audit row: `id, study_id, user, decision, comment, model_version, confidence_at_time, timestamp`.
