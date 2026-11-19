import { useCallback, useEffect, useState } from "react";
import Dashboard from "./Dashboard.jsx";
import { decide, getAudit, isLoggedIn, listStudies, login, logout } from "./api.js";

//const DISCLAIMER = "FOR EDUCATION ONLY. NOT FOR CLINICAL USE.";
const STATUS_LABEL = { PENDING_REVIEW: "Pending", APPROVED: "Approved", REJECTED: "Rejected" };
const fmt = (iso) => {
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
};

function Logo() {
  return (
    <div className="logo" aria-hidden="true">
      <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
        <path d="M12 4v16M4 12h16" />
      </svg>
    </div>
  );
}

//function Disclaimer() {
// return <div className="disclaimer">{DISCLAIMER}</div>;
//}

function Login({ onDone }) {
  const [u, setU] = useState("radiologist");
  const [p, setP] = useState("");
  const [err, setErr] = useState("");
  const submit = async (e) => {
    e.preventDefault();
    setErr("");
    try {
      await login(u, p);
      onDone();
    } catch {
      setErr("Incorrect username or password.");
    }
  };
  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <div className="brand-lg"><Logo /><div><h1>CXR Review</h1><span>AI-assisted chest X-ray workstation</span></div></div>
        <label>Username<input value={u} onChange={(e) => setU(e.target.value)} autoComplete="username" /></label>
        <label>Password<input type="password" value={p} onChange={(e) => setP(e.target.value)} autoComplete="current-password" /></label>
        {err && <div className="form-err">{err}</div>}
        <button className="btn primary" type="submit">Sign in</button>
        <p className="fine">Radiologist role required. Decisions are recorded in an append-only audit log.</p>
      </form>
    </div>
  );
}

function Gauge({ value, tone }) {
  const r = 52, c = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 120 120" className="gauge" role="img" aria-label={`Model score ${Math.round(value * 100)} percent`}>
      <circle cx="60" cy="60" r={r} className="g-track" />
      <circle cx="60" cy="60" r={r} className={`g-fill ${tone}`} strokeDasharray={`${c * value} ${c}`} transform="rotate(-90 60 60)" />
      <text x="60" y="62" textAnchor="middle" className="g-num">{Math.round(value * 100)}</text>
      <text x="60" y="80" textAnchor="middle" className="g-unit">% score</text>
    </svg>
  );
}

function Viewer({ s, onDecided }) {
  const [heat, setHeat] = useState(true);
  const [mask, setMask] = useState(false);
  const [opacity, setOpacity] = useState(0.6);
  const [zoom, setZoom] = useState(1);
  const [bright, setBright] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [invert, setInvert] = useState(false);
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);

  const loadHistory = useCallback(async () => {
    try {
      const rows = await getAudit();
      setHistory(rows.filter((r) => r.study_id === s.study_id));
    } catch { /* ignore */ }
  }, [s.study_id]);
  useEffect(() => { loadHistory(); }, [loadHistory]);

  const send = async (d) => {
    setBusy(true);
    setError("");
    try {
      await decide(s.study_id, d, comment);
      setComment("");
      await loadHistory();
      onDecided();
    } catch (e) {
      setError("Could not save the decision: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const tone = s.uncertain ? "amber" : s.predicted_label ? "red" : "green";
  const verdict = s.predicted_label ? "Lung opacity suspected" : "No lung opacity suspected";
  const baseFilter = `brightness(${bright}%) contrast(${contrast}%) ${invert ? "invert(1)" : ""}`;
  const reset = () => { setZoom(1); setBright(100); setContrast(100); setInvert(false); };

  return (
    <div className="workspace">
      <div className="viewport-col">
        <div className="toolbar">
          <div className="group">
            <button className={`chip ${heat ? "on" : ""}`} onClick={() => setHeat(!heat)}>Grad-CAM heatmap</button>
            <button className={`chip ${mask ? "on" : ""}`} onClick={() => setMask(!mask)}>Segmentation</button>
          </div>
          <label className="slider">Overlay
            <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} />
          </label>
          <label className="slider">Brightness
            <input type="range" min="50" max="150" value={bright} onChange={(e) => setBright(+e.target.value)} />
          </label>
          <label className="slider">Contrast
            <input type="range" min="50" max="200" value={contrast} onChange={(e) => setContrast(+e.target.value)} />
          </label>
          <div className="group">
            <button className="icon" onClick={() => setZoom(Math.max(1, +(zoom - 0.25).toFixed(2)))} title="Zoom out">-</button>
            <span className="zoomval">{Math.round(zoom * 100)}%</span>
            <button className="icon" onClick={() => setZoom(Math.min(3, +(zoom + 0.25).toFixed(2)))} title="Zoom in">+</button>
            <button className={`chip ${invert ? "on" : ""}`} onClick={() => setInvert(!invert)}>Invert</button>
            <button className="chip" onClick={reset}>Reset</button>
          </div>
        </div>
        <div className="viewport">
          <div className="canvas" style={{ transform: `scale(${zoom})` }}>
            <img src={s.urls.image} alt="Chest X-ray" style={{ filter: baseFilter }} />
            {heat && <img className="overlay" style={{ opacity }} src={s.urls.heatmap} alt="Grad-CAM heatmap" />}
            {mask && <img className="overlay" style={{ opacity }} src={s.urls.mask} alt="Segmentation mask" />}
          </div>
          <div className="hud tl"><b>Study {s.study_id.slice(0, 8)}</b><span>{s.description || "Chest X-ray"}</span></div>
          <div className="hud tr"><span>Model {s.model_version}</span><span>Received {fmt(s.created_at)}</span></div>
          <div className="hud bl">
            {heat && <span><i className="dot heat" /> Grad-CAM: regions that raised the score</span>}
            {mask && <span><i className="dot mask" /> Segmentation (box-derived training masks)</span>}
          </div>

        </div>
      </div>

      <aside className="side">
        <section className="card">
          <h3>AI finding</h3>
          <div className="finding">
            <Gauge value={s.probability} tone={tone} />
            <div>
              <div className={`verdict ${tone}`}>{verdict}</div>
              <div className="muted small">Decision threshold {Math.round(s.threshold * 100)}%</div>
            </div>
          </div>
          {s.uncertain && <div className="notice amber">Low confidence (40-60%). Review carefully.</div>}
          {s.untrained && <div className="notice red">Untrained demo weights: this score is meaningless.</div>}
          <p className="fine">Suggestion only. The radiologist decides.</p>
        </section>

        <section className="card">
          <h3>Radiologist decision</h3>
          <div className="status-line">Status <span className={`pill ${s.status}`}>{STATUS_LABEL[s.status] || s.status}</span></div>
          <textarea placeholder="Comment (optional)" value={comment} onChange={(e) => setComment(e.target.value)} maxLength={2000} />
          {error && <div className="notice red">{error}</div>}
          <div className="actions">
            <button className="btn approve" disabled={busy} onClick={() => send("approve")}>Approve AI finding</button>
            <button className="btn reject" disabled={busy} onClick={() => send("reject")}>Reject</button>
          </div>
        </section>

        <section className="card">
          <h3>History for this study</h3>
          {history.length === 0 && <p className="muted small">No decisions recorded yet.</p>}
          <ul className="history">
            {history.map((r) => (
              <li key={r.id}>
                <span className={`pill ${r.decision === "approve" ? "APPROVED" : "REJECTED"}`}>{r.decision}</span>
                <span>{r.user}</span>
                <span className="muted small">{fmt(r.timestamp)}</span>
                {r.comment && <div className="small">{r.comment}</div>}
              </li>
            ))}
          </ul>
        </section>
      </aside>
    </div>
  );
}

function Worklist({ studies, sel, onSel, filter, setFilter }) {
  const counts = {
    ALL: studies.length,
    PENDING_REVIEW: studies.filter((s) => s.status === "PENDING_REVIEW").length,
    APPROVED: studies.filter((s) => s.status === "APPROVED").length,
    REJECTED: studies.filter((s) => s.status === "REJECTED").length,
  };
  const shown = filter === "ALL" ? studies : studies.filter((s) => s.status === filter);
  return (
    <aside className="worklist">
      <div className="filters">
        {[["ALL", "All"], ["PENDING_REVIEW", "Pending"], ["APPROVED", "Approved"], ["REJECTED", "Rejected"]].map(([k, l]) => (
          <button key={k} className={`chip ${filter === k ? "on" : ""}`} onClick={() => setFilter(k)}>{l} <em>{counts[k]}</em></button>
        ))}
      </div>
      <div className="rows">
        {shown.length === 0 && <p className="empty">No studies here yet. Upload a DICOM to Orthanc and it appears within seconds.</p>}
        {shown.map((s) => (
          <button key={s.study_id} className={`row ${sel === s.study_id ? "sel" : ""}`} onClick={() => onSel(s.study_id)}>
            <img src={s.urls.image} alt="" loading="lazy" />
            <div className="meta">
              <b>{s.study_id.slice(0, 8)}</b>
              <span className="muted small">{fmt(s.created_at)}</span>
              <span className={`pill ${s.status}`}>{STATUS_LABEL[s.status] || s.status}</span>
            </div>
            <div className={`score ${s.uncertain ? "amber" : s.predicted_label ? "red" : "green"}`}>{Math.round(s.probability * 100)}</div>
          </button>
        ))}
      </div>
    </aside>
  );
}

function Audit({ rows }) {
  return (
    <div className="audit">
      <h2>Audit log</h2>
      <p className="muted">Append-only record of every radiologist decision. Rows cannot be edited or deleted.</p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>#</th><th>Time</th><th>User</th><th>Study</th><th>Decision</th><th>AI score</th><th>Model</th><th>Comment</th></tr>
          </thead>
          <tbody>
            {rows.length === 0 && <tr><td colSpan="8" className="muted">No decisions yet.</td></tr>}
            {rows.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td><td>{fmt(r.timestamp)}</td><td>{r.user}</td><td>{r.study_id.slice(0, 8)}</td>
                <td><span className={`pill ${r.decision === "approve" ? "APPROVED" : "REJECTED"}`}>{r.decision}</span></td>
                <td>{Math.round(r.confidence_at_time * 100)}%</td><td>{r.model_version}</td><td>{r.comment}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function App() {
  const [authed, setAuthed] = useState(isLoggedIn());
  const [studies, setStudies] = useState([]);
  const [sel, setSel] = useState(null);
  const [tab, setTab] = useState("dashboard");
  const [filter, setFilter] = useState("ALL");
  const [audit, setAudit] = useState([]);

  const refresh = useCallback(async () => {
    try {
      const list = await listStudies();
      setStudies(list);
      setSel((cur) => cur || (list[0] ? list[0].study_id : null));
      if (tab === "audit") setAudit(await getAudit());
    } catch (e) {
      if (e.message === "auth") setAuthed(false);
    }
  }, [tab]);

  useEffect(() => {
    if (!authed) return undefined;
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [authed, refresh]);

  if (!authed) return (<><Disclaimer /><Login onDone={() => setAuthed(true)} /></>);
  const current = studies.find((s) => s.study_id === sel) || null;

  return (
    <div className="app">
      <Disclaimer />
      <header className="topbar">
        <div className="brand"><Logo /><div><b>CXR Review</b><span>AI-assisted chest X-ray workstation</span></div></div>
        <nav>
          <button className={tab === "dashboard" ? "on" : ""} onClick={() => setTab("dashboard")}>Dashboard</button>
          <button className={tab === "review" ? "on" : ""} onClick={() => setTab("review")}>Worklist</button>
          <button className={tab === "audit" ? "on" : ""} onClick={() => { setTab("audit"); getAudit().then(setAudit).catch(() => { }); }}>Audit log</button>
        </nav>
        <button className="btn ghost" onClick={() => { logout(); setAuthed(false); }}>Sign out</button>
      </header>
      {tab === "dashboard" ? (
        <Dashboard studies={studies} onOpen={(id) => { setSel(id); setTab("review"); }} />
      ) : tab === "audit" ? (
        <Audit rows={audit} />
      ) : (
        <div className="main">
          <Worklist studies={studies} sel={sel} onSel={setSel} filter={filter} setFilter={setFilter} />
          {current ? <Viewer key={current.study_id} s={current} onDecided={refresh} /> : <div className="placeholder">Select a study from the worklist.</div>}
        </div>
      )}
    </div>
  );
}
