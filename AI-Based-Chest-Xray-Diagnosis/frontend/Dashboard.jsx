import { useEffect, useState } from "react";
import { getAudit } from "./api.js";

const pct = (n, d) => (d ? Math.round((n / d) * 100) : 0);
const fmt = (iso) => {
    try { return new Date(iso).toLocaleString(); } catch { return iso; }
};

function Kpi({ label, value, sub, tone }) {
    return (
        <div className={`kpi ${tone || ""}`}>
            <span className="kpi-label">{label}</span>
            <b className="kpi-value">{value}</b>
            <span className="kpi-sub">{sub}</span>
        </div>
    );
}

function Donut({ parts }) {
    const total = parts.reduce((a, p) => a + p.value, 0);
    const r = 42, c = 2 * Math.PI * r;
    let offset = 0;
    return (
        <div className="donut-wrap">
            <svg viewBox="0 0 120 120" className="donut" role="img" aria-label="Studies by status">
                <circle cx="60" cy="60" r={r} className="g-track" />
                {total > 0 && parts.map((p) => {
                    const len = (p.value / total) * c;
                    const el = (
                        <circle key={p.label} cx="60" cy="60" r={r} fill="none" strokeWidth="14"
                            stroke={p.color} strokeDasharray={`${len} ${c - len}`} strokeDashoffset={-offset}
                            transform="rotate(-90 60 60)" />
                    );
                    offset += len;
                    return el;
                })}
                <text x="60" y="64" textAnchor="middle" className="g-num">{total}</text>
            </svg>
            <ul className="legend">
                {parts.map((p) => (
                    <li key={p.label}><i style={{ background: p.color }} />{p.label}<b>{p.value}</b></li>
                ))}
            </ul>
        </div>
    );
}

function Bars({ items }) {
    const max = Math.max(1, ...items.map((i) => i.value));
    return (
        <div className="bars">
            {items.map((i) => (
                <div className="bar-col" key={i.label} title={`${i.label}: ${i.value}`}>
                    <span className="bar-num">{i.value || ""}</span>
                    <div className="bar-fill" style={{ height: `${(i.value / max) * 100}%`, background: i.color || "var(--accent)" }} />
                    <span className="bar-label">{i.label}</span>
                </div>
            ))}
        </div>
    );
}

export default function Dashboard({ studies, onOpen }) {
    const [audit, setAudit] = useState([]);
    useEffect(() => {
        let alive = true;
        const load = () => getAudit().then((r) => alive && setAudit(r)).catch(() => { });
        load();
        const t = setInterval(load, 10000);
        return () => { alive = false; clearInterval(t); };
    }, []);

    const total = studies.length;
    const pending = studies.filter((s) => s.status === "PENDING_REVIEW");
    const approved = studies.filter((s) => s.status === "APPROVED").length;
    const rejected = studies.filter((s) => s.status === "REJECTED").length;
    const positives = studies.filter((s) => s.predicted_label === 1).length;
    const uncertain = studies.filter((s) => s.uncertain).length;
    const avg = total ? Math.round((studies.reduce((a, s) => a + s.probability, 0) / total) * 100) : 0;
    const decided = approved + rejected;

    // score histogram: 10 bins; the 40-60% band is amber (low confidence)
    const bins = Array.from({ length: 10 }, (_, i) => ({ label: `${i * 10}`, value: 0, color: i < 4 ? "var(--green)" : i < 6 ? "var(--amber)" : "var(--red)" }));
    studies.forEach((s) => { bins[Math.min(9, Math.floor(s.probability * 10))].value += 1; });

    // studies received per day, last 7 days
    const days = Array.from({ length: 7 }, (_, k) => {
        const d = new Date();
        d.setDate(d.getDate() - (6 - k));
        return { key: d.toDateString(), label: d.toLocaleDateString(undefined, { weekday: "short" }), value: 0 };
    });
    studies.forEach((s) => {
        const hit = days.find((d) => d.key === new Date(s.created_at).toDateString());
        if (hit) hit.value += 1;
    });

    const attention = [...pending].sort((a, b) => Math.abs(a.probability - 0.5) - Math.abs(b.probability - 0.5)).slice(0, 6);

    return (
        <div className="dash">
            <div className="dash-head">
                <div>
                    <h2>Dashboard</h2>
                    <p className="muted">Live overview of AI-assisted chest X-ray review. Updates automatically.</p>
                </div>
            </div>

            {total === 0 && (
                <div className="notice amber" style={{ marginBottom: 16 }}>
                    No studies yet. Upload DICOM files to Orthanc and they will appear here within about 15 seconds.
                </div>
            )}

            <div className="kpis">
                <Kpi label="Total studies" value={total} sub="processed by the AI" />
                <Kpi label="Pending review" value={pending.length} sub="waiting for a radiologist" tone={pending.length ? "amber" : ""} />
                <Kpi label="Approved" value={approved} sub={`${pct(approved, decided)}% of decided`} tone="green" />
                <Kpi label="Rejected" value={rejected} sub={`${pct(rejected, decided)}% of decided`} tone="red" />
                <Kpi label="AI flagged opacity" value={`${pct(positives, total)}%`} sub={`${positives} of ${total} studies`} />
                <Kpi label="Low confidence" value={uncertain} sub="scores between 40 and 60%" tone={uncertain ? "amber" : ""} />
                <Kpi label="Average AI score" value={`${avg}%`} sub="across all studies" />
                <Kpi label="Radiologist agreement" value={decided ? `${pct(approved, decided)}%` : "-"} sub="approved vs. rejected AI findings" />
            </div>

            <div className="charts">
                <section className="card">
                    <h3>Review status</h3>
                    <Donut parts={[
                        { label: "Pending", value: pending.length, color: "var(--amber)" },
                        { label: "Approved", value: approved, color: "var(--green)" },
                        { label: "Rejected", value: rejected, color: "var(--red)" },
                    ]} />
                </section>
                <section className="card">
                    <h3>AI score distribution (%)</h3>
                    <Bars items={bins} />
                </section>
                <section className="card">
                    <h3>Studies received, last 7 days</h3>
                    <Bars items={days} />
                </section>
            </div>

            <div className="panels">
                <section className="card">
                    <h3>Needs attention first</h3>
                    <p className="muted small" style={{ marginTop: 0 }}>Pending studies, most uncertain AI scores on top.</p>
                    {attention.length === 0 && <p className="muted small">Nothing waiting for review.</p>}
                    <ul className="attn">
                        {attention.map((s) => (
                            <li key={s.study_id}>
                                <button onClick={() => onOpen(s.study_id)}>
                                    <img src={s.urls.image} alt="" loading="lazy" />
                                    <span><b>{s.study_id.slice(0, 8)}</b><em className="muted small">{fmt(s.created_at)}</em></span>
                                    <strong className={s.uncertain ? "amber" : s.predicted_label ? "red" : "green"}>{Math.round(s.probability * 100)}%</strong>
                                </button>
                            </li>
                        ))}
                    </ul>
                </section>
                <section className="card">
                    <h3>Recent decisions</h3>
                    {audit.length === 0 && <p className="muted small">No decisions recorded yet.</p>}
                    <ul className="history">
                        {audit.slice(0, 6).map((r) => (
                            <li key={r.id}>
                                <span className={`pill ${r.decision === "approve" ? "APPROVED" : "REJECTED"}`}>{r.decision}</span>
                                <span>{r.study_id.slice(0, 8)}</span>
                                <span className="muted small">{r.user} - {fmt(r.timestamp)}</span>
                            </li>
                        ))}
                    </ul>
                </section>
            </div>
            <p className="fine" style={{ marginTop: 16 }}>FOR EDUCATION ONLY. NOT FOR CLINICAL USE.</p>
        </div>
    );
}
