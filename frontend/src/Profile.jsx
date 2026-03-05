import { useState, useEffect } from "react";

const API = "http://127.0.0.1:8000";

export default function Profile({ user, onBack }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const [expanded, setExpanded] = useState(null); // index de la carte ouverte

  // ── charger le profil au montage ──
  useEffect(() => {
    fetch(`${API}/profile`, {
      headers: { "Authorization": `Bearer ${user.access_token}` }
    })
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => { setError("Could not load profile."); setLoading(false); });
  }, []);

  // ── initiales de l'avatar ──
  const initials = user.username
    .split(" ")
    .map(w => w[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  if (loading) return (
    <div style={s.page}>
      <div style={s.center}>
        <div style={s.spinnerRing} />
        <p style={{ color: "#a8a29e", marginTop: 16 }}>Loading your profile…</p>
      </div>
    </div>
  );

  if (error) return (
    <div style={s.page}>
      <div style={s.center}>
        <p style={{ color: "#be123c" }}>{error}</p>
        <button style={s.backBtn} onClick={onBack}>← Back</button>
      </div>
    </div>
  );

  const { username, email, member_since, stats, history } = data;
  const memberDate = member_since !== "N/A"
    ? new Date(member_since).toLocaleDateString("en-GB", { year: "numeric", month: "long", day: "numeric" })
    : "N/A";

  return (
    <div style={s.page}>

      {/* ── NAV ── */}
      <nav style={s.nav}>
        <div style={s.navLogo}><span style={s.navDot} />JobScan</div>
        <button style={s.backBtn} onClick={onBack}>← Back to App</button>
      </nav>

      <div style={s.content}>

        {/* ── HERO CARD ── */}
        <div style={s.heroCard}>
          <div style={s.avatar}>{initials}</div>
          <div>
            <h1 style={s.heroName}>{username}</h1>
            <p style={s.heroEmail}>{email}</p>
            <p style={s.heroDate}>Member since {memberDate}</p>
          </div>
        </div>

        {/* ── STATS ROW ── */}
        <div style={s.statsRow}>
          <div style={s.statCard}>
            <span style={s.statNum}>{stats.total_analyses}</span>
            <span style={s.statLabel}>CVs Analysed</span>
          </div>
          <div style={s.statCard}>
            <span style={{ ...s.statNum, fontSize: 18 }}>{stats.top_job || "—"}</span>
            <span style={s.statLabel}>Top Prediction</span>
          </div>
          <div style={s.statCard}>
            <span style={s.statNum}>{stats.avg_confidence}%</span>
            <span style={s.statLabel}>Avg Confidence</span>
          </div>
        </div>

        {/* ── HISTORY ── */}
        <div style={s.section}>
          <h2 style={s.sectionTitle}>📋 Prediction History</h2>

          {history.length === 0 ? (
            <div style={s.emptyBox}>
              <p style={{ fontSize: 32, marginBottom: 12 }}>📂</p>
              <p style={{ color: "#a8a29e", fontSize: 14 }}>No analyses yet. Upload a CV to get started!</p>
            </div>
          ) : (
            <div style={s.historyList}>
              {history.map((item, i) => (
                <div key={i} style={s.historyCard}>

                  {/* ── header de la carte ── */}
                  <div style={s.historyHeader} onClick={() => setExpanded(expanded === i ? null : i)}>
                    <div style={s.historyLeft}>
                      <div style={s.historyJobBadge}>{item.predicted_job}</div>
                      <div style={s.historyMeta}>
                        <span style={s.historyFile}>📄 {item.filename}</span>
                        <span style={s.historyDate}>🕐 {item.date}</span>
                      </div>
                    </div>
                    <div style={s.historyRight}>
                      <div style={s.confBadge}>
                        {(item.confidence * 100).toFixed(1)}%
                      </div>
                      <span style={s.chevron}>{expanded === i ? "▲" : "▼"}</span>
                    </div>
                  </div>

                  {/* ── détails dépliables ── */}
                  {expanded === i && (
                    <div style={s.historyDetails}>

                      {/* top 3 */}
                      <p style={s.detailTitle}>Top 3 Predictions</p>
                      {item.top_3.map((t, j) => (
                        <div key={j} style={s.barRow}>
                          <span style={s.barLabel}>{t.job}</span>
                          <div style={s.barTrack}>
                            <div style={{
                              ...s.barFill,
                              width: `${(t.probability / item.top_3[0].probability) * 100}%`,
                              background: ["#f97316","#fb923c","#fdba74"][j],
                            }} />
                          </div>
                          <span style={s.barVal}>{(t.probability * 100).toFixed(1)}%</span>
                        </div>
                      ))}

                      {/* skills */}
                      {item.skills && (
                        <>
                          <p style={{ ...s.detailTitle, marginTop: 16 }}>🛠 Skills detected</p>
                          <div style={s.tags}>
                            {item.skills.split(";").filter(Boolean).map((sk, k) => (
                              <span key={k} style={s.tag}>{sk}</span>
                            ))}
                          </div>
                        </>
                      )}

                      {/* certifications */}
                      {item.certifications && (
                        <>
                          <p style={{ ...s.detailTitle, marginTop: 12 }}>🏅 Certifications</p>
                          <div style={s.tags}>
                            {item.certifications.split(";").filter(Boolean).map((c, k) => (
                              <span key={k} style={{ ...s.tag, background: "#fff7ed", color: "#c2410c" }}>{c}</span>
                            ))}
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
      `}</style>
    </div>
  );
}

/* ── STYLES ── */
const s = {
  page:    { minHeight: "100vh", background: "#fffbf7", fontFamily: "'Segoe UI', sans-serif", color: "#1c1917" },
  center:  { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "80vh", gap: 16 },

  /* nav */
  nav:     { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "20px 60px", borderBottom: "1px solid #f5e6d8" },
  navLogo: { display: "flex", alignItems: "center", gap: 8, fontWeight: 800, fontSize: 20, color: "#f97316" },
  navDot:  { width: 10, height: 10, borderRadius: "50%", background: "#f97316" },
  backBtn: { background: "none", border: "1px solid #f5e6d8", color: "#78716c", borderRadius: 8, padding: "7px 18px", cursor: "pointer", fontSize: 13, fontWeight: 600 },

  /* spinner */
  spinnerRing: { width: 40, height: 40, border: "4px solid #f5e6d8", borderTop: "4px solid #f97316", borderRadius: "50%", animation: "spin .8s linear infinite" },

  /* content */
  content: { maxWidth: 720, margin: "0 auto", padding: "40px 24px 60px" },

  /* hero */
  heroCard:  { display: "flex", alignItems: "center", gap: 24, background: "#fff", border: "1px solid #f5e6d8", borderRadius: 20, padding: "28px 32px", marginBottom: 24, boxShadow: "0 4px 24px rgba(249,115,22,.07)" },
  avatar:    { width: 72, height: 72, borderRadius: "50%", background: "#f97316", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 26, fontWeight: 900, flexShrink: 0 },
  heroName:  { fontSize: 24, fontWeight: 900, color: "#1c1917", marginBottom: 4 },
  heroEmail: { fontSize: 14, color: "#78716c", marginBottom: 4 },
  heroDate:  { fontSize: 12, color: "#a8a29e" },

  /* stats */
  statsRow: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 32 },
  statCard: { background: "#fff", border: "1px solid #f5e6d8", borderRadius: 16, padding: "20px 16px", textAlign: "center", boxShadow: "0 2px 12px rgba(249,115,22,.05)" },
  statNum:  { display: "block", fontSize: 28, fontWeight: 900, color: "#f97316", marginBottom: 6 },
  statLabel:{ display: "block", fontSize: 12, color: "#a8a29e", fontWeight: 600, textTransform: "uppercase", letterSpacing: .6 },

  /* section */
  section:      { },
  sectionTitle: { fontSize: 18, fontWeight: 800, color: "#1c1917", marginBottom: 16 },

  /* empty */
  emptyBox: { background: "#fff", border: "1px solid #f5e6d8", borderRadius: 16, padding: "48px 24px", textAlign: "center" },

  /* history */
  historyList: { display: "flex", flexDirection: "column", gap: 12 },
  historyCard: { background: "#fff", border: "1px solid #f5e6d8", borderRadius: 16, overflow: "hidden", boxShadow: "0 2px 12px rgba(249,115,22,.04)" },

  historyHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "16px 20px", cursor: "pointer" },
  historyLeft:   { display: "flex", flexDirection: "column", gap: 6 },
  historyRight:  { display: "flex", alignItems: "center", gap: 12, flexShrink: 0 },

  historyJobBadge: { fontWeight: 800, fontSize: 15, color: "#1c1917" },
  historyMeta:     { display: "flex", gap: 16 },
  historyFile:     { fontSize: 12, color: "#a8a29e" },
  historyDate:     { fontSize: 12, color: "#a8a29e" },

  confBadge: { background: "#fff7ed", color: "#f97316", fontWeight: 800, fontSize: 13, padding: "4px 10px", borderRadius: 20 },
  chevron:   { color: "#a8a29e", fontSize: 11 },

  /* details */
  historyDetails: { borderTop: "1px solid #f5e6d8", padding: "16px 20px", background: "#fffbf7" },
  detailTitle:    { fontSize: 11, color: "#a8a29e", fontWeight: 700, textTransform: "uppercase", letterSpacing: .8, marginBottom: 10 },

  barRow:   { display: "flex", alignItems: "center", gap: 10, marginBottom: 8 },
  barLabel: { fontSize: 12, color: "#44403c", width: 160, flexShrink: 0 },
  barTrack: { flex: 1, height: 7, background: "#f5f5f4", borderRadius: 4, overflow: "hidden" },
  barFill:  { height: "100%", borderRadius: 4 },
  barVal:   { fontSize: 11, color: "#a8a29e", width: 36, textAlign: "right" },

  tags: { display: "flex", flexWrap: "wrap", gap: 6 },
  tag:  { padding: "3px 10px", borderRadius: 20, background: "#f5f5f4", color: "#57534e", fontSize: 11, fontWeight: 600 },
};