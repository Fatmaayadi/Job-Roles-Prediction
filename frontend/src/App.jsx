import { useState } from "react";

const API = "http://127.0.0.1:8000";

export default function App() {
  const [file, setFile]         = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState(null);

  const onDrop = (e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) { setFile(f); setResult(null); setError(null); }
  };
  const onFileChange = (e) => {
    const f = e.target.files[0];
    if (f) { setFile(f); setResult(null); setError(null); }
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res  = await fetch(`${API}/predict-cv`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Server error");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const maxProb = result ? result.top_3[0].probability : 1;
  const colors  = ["#f97316", "#fb923c", "#fdba74"];

  return (
    <div style={s.page}>

      {/* ── NAV ── */}
      <nav style={s.nav}>
        <div style={s.navLogo}>
          <span style={s.navDot} />
          JobScan
        </div>
        <div style={s.navLinks}>
          <a href="#" style={s.navLink}>Home</a>
          <a href="#" style={s.navLink}>About</a>
          <a href="#" style={s.navLink}>Contact</a>
        </div>
      </nav>

      {/* ── HERO ── */}
      <section style={s.hero}>
        {/* floating blobs */}
        <div style={{ ...s.blob, top: 60,  left: "10%", width: 320, height: 320, background: "#fff7ed", animationDelay: "0s"   }} />
        <div style={{ ...s.blob, top: 200, right: "8%", width: 240, height: 240, background: "#ffedd5", animationDelay: "2s"   }} />
        <div style={{ ...s.blob, bottom: 0, left: "30%", width: 180, height: 180, background: "#fed7aa", animationDelay: "4s" }} />

        <div style={s.heroContent}>
          {/* icon */}
          <div style={s.iconWrap}>
            <svg width="38" height="38" viewBox="0 0 24 24" fill="none">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"
                stroke="#f97316" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="9"  cy="10" r="1" fill="#f97316"/>
              <circle cx="12" cy="10" r="1" fill="#f97316"/>
              <circle cx="15" cy="10" r="1" fill="#f97316"/>
            </svg>
          </div>

          <p style={s.greeting}>Hello there 👋 — We're pleased to welcome you.</p>
          <h1 style={s.heroTitle}>
            Discover your<br />
            <span style={s.heroAccent}>perfect job role</span>
          </h1>
          <p style={s.heroSub}>
            Upload your CV and let our AI instantly match your skills<br />
            to the career path you were made for.
          </p>

          {/* ── UPLOAD ZONE ── */}
          <div
            style={{ ...s.dropzone, ...(dragging ? s.dropzoneActive : {}) }}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            {file ? (
              <div style={s.fileChosen}>
                <span style={s.fileIcon}>📄</span>
                <span style={s.fileName}>{file.name}</span>
                <label style={s.changeBtn}>
                  Change
                  <input type="file" accept=".pdf,.txt" onChange={onFileChange} style={{ display: "none" }} />
                </label>
              </div>
            ) : (
              <>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style={{ marginBottom: 10 }}>
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="#f97316" strokeWidth="2" strokeLinecap="round"/>
                  <polyline points="17 8 12 3 7 8"  stroke="#f97316" strokeWidth="2" strokeLinecap="round"/>
                  <line x1="12" y1="3" x2="12" y2="15" stroke="#f97316" strokeWidth="2" strokeLinecap="round"/>
                </svg>
                <p style={s.dropText}>Drag & drop your CV here</p>
                <p style={s.dropSub}>PDF or TXT accepted</p>
                <label style={s.browseBtn}>
                  Browse file
                  <input type="file" accept=".pdf,.txt" onChange={onFileChange} style={{ display: "none" }} />
                </label>
              </>
            )}
          </div>

          <button
            style={{ ...s.predictBtn, ...(loading || !file ? s.predictBtnOff : {}) }}
            onClick={handleSubmit}
            disabled={loading || !file}
          >
            {loading ? (
              <span style={s.spinner}>⏳ Analysing your CV…</span>
            ) : (
              "Predict My Job Role →"
            )}
          </button>

          {/* ── ERROR ── */}
          {error && <div style={s.errorBox}>⚠️ {error}</div>}
        </div>
      </section>

      {/* ── RESULTS ── */}
      {result && (
        <section style={s.resultSection}>
          <div style={s.resultCard}>
            <p style={s.resultLabel}>✨ Best Match</p>
            <h2 style={s.resultJob}>{result.predicted_job}</h2>
            <p style={s.resultConf}>{(result.confidence * 100).toFixed(1)}% confidence</p>

            {result.extracted?.skills && (
              <div style={s.extractedBox}>
                <p style={s.extractedTitle}>🛠 Skills detected</p>
                <div style={s.tags}>
                  {result.extracted.skills.split(";").map((sk, i) => (
                    <span key={i} style={s.tag}>{sk}</span>
                  ))}
                </div>
              </div>
            )}

            {result.extracted?.certifications && (
              <div style={s.extractedBox}>
                <p style={s.extractedTitle}>🏅 Certifications detected</p>
                <div style={s.tags}>
                  {result.extracted.certifications.split(";").map((c, i) => (
                    <span key={i} style={{ ...s.tag, background: "#fff7ed", color: "#c2410c" }}>{c}</span>
                  ))}
                </div>
              </div>
            )}

            <p style={s.top3Title}>Top 3 Predictions</p>
            {result.top_3.map((item, i) => (
              <div key={i} style={s.barRow}>
                <span style={s.barLabel}>{item.job}</span>
                <div style={s.barTrack}>
                  <div style={{
                    ...s.barFill,
                    width: `${(item.probability / maxProb) * 100}%`,
                    background: colors[i],
                  }} />
                </div>
                <span style={s.barVal}>{(item.probability * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── FOOTER ── */}
      <footer style={s.footer}>
        <p>© 2026 JobScan · Built with ❤️ and Machine Learning</p>
      </footer>

      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px) scale(1); }
          50%       { transform: translateY(-20px) scale(1.04); }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #fffbf7; }
        a { text-decoration: none; }
      `}</style>
    </div>
  );
}

/* ── STYLES ── */
const s = {
  page:  { minHeight: "100vh", fontFamily: "'Segoe UI', sans-serif", background: "#fffbf7", color: "#1c1917" },

  /* nav */
  nav:      { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "20px 60px", borderBottom: "1px solid #f5e6d8" },
  navLogo:  { display: "flex", alignItems: "center", gap: 8, fontWeight: 800, fontSize: 20, color: "#f97316" },
  navDot:   { width: 10, height: 10, borderRadius: "50%", background: "#f97316", display: "inline-block" },
  navLinks: { display: "flex", gap: 32 },
  navLink:  { color: "#78716c", fontSize: 14, fontWeight: 500, transition: "color .2s" },

  /* hero */
  hero:        { position: "relative", overflow: "hidden", padding: "80px 24px 60px", textAlign: "center" },
  blob:        { position: "absolute", borderRadius: "50%", filter: "blur(60px)", opacity: 0.6, animation: "float 6s ease-in-out infinite" },
  heroContent: { position: "relative", zIndex: 1, maxWidth: 600, margin: "0 auto" },

  iconWrap:  { display: "inline-flex", alignItems: "center", justifyContent: "center", width: 70, height: 70, borderRadius: 20, background: "#fff7ed", border: "2px solid #fed7aa", marginBottom: 24 },
  greeting:  { fontSize: 14, color: "#a8a29e", fontWeight: 500, marginBottom: 16, letterSpacing: .5 },
  heroTitle: { fontSize: 46, fontWeight: 900, lineHeight: 1.15, color: "#1c1917", marginBottom: 18 },
  heroAccent:{ color: "#f97316" },
  heroSub:   { fontSize: 16, color: "#78716c", lineHeight: 1.7, marginBottom: 40 },

  /* dropzone */
  dropzone:       { border: "2px dashed #fed7aa", borderRadius: 16, padding: "36px 24px", background: "#fff7ed", cursor: "pointer", transition: "all .2s", marginBottom: 20 },
  dropzoneActive: { borderColor: "#f97316", background: "#ffedd5" },
  dropText:       { fontWeight: 600, fontSize: 15, color: "#1c1917", marginBottom: 6 },
  dropSub:        { fontSize: 13, color: "#a8a29e", marginBottom: 16 },
  browseBtn:      { display: "inline-block", padding: "9px 22px", borderRadius: 8, background: "#f97316", color: "#fff", fontSize: 13, fontWeight: 700, cursor: "pointer" },
  fileChosen:     { display: "flex", alignItems: "center", justifyContent: "center", gap: 12 },
  fileIcon:       { fontSize: 24 },
  fileName:       { fontWeight: 600, fontSize: 14, color: "#1c1917" },
  changeBtn:      { fontSize: 12, color: "#f97316", fontWeight: 700, cursor: "pointer", textDecoration: "underline" },

  /* predict button */
  predictBtn:    { width: "100%", padding: "16px", borderRadius: 12, background: "#1c1917", color: "#fff", fontWeight: 800, fontSize: 16, border: "none", cursor: "pointer", transition: "opacity .2s" },
  predictBtnOff: { opacity: 0.3, cursor: "not-allowed" },
  spinner:       { display: "flex", alignItems: "center", justifyContent: "center", gap: 8 },

  errorBox: { marginTop: 16, padding: "12px 16px", borderRadius: 10, background: "#fff1f2", border: "1px solid #fda4af", color: "#be123c", fontSize: 14 },

  /* results */
  resultSection: { padding: "0 24px 60px", display: "flex", justifyContent: "center" },
  resultCard:    { width: "100%", maxWidth: 560, background: "#fff", border: "1px solid #f5e6d8", borderRadius: 20, padding: 32, boxShadow: "0 8px 40px rgba(249,115,22,.08)" },
  resultLabel:   { fontSize: 12, letterSpacing: 2, color: "#f97316", textTransform: "uppercase", marginBottom: 8 },
  resultJob:     { fontSize: 30, fontWeight: 900, color: "#1c1917", marginBottom: 4 },
  resultConf:    { fontSize: 14, color: "#a8a29e", marginBottom: 24 },

  extractedBox:   { background: "#fffbf7", border: "1px solid #f5e6d8", borderRadius: 12, padding: "14px 16px", marginBottom: 16 },
  extractedTitle: { fontSize: 12, color: "#a8a29e", fontWeight: 600, marginBottom: 10 },
  tags:           { display: "flex", flexWrap: "wrap", gap: 8 },
  tag:            { padding: "4px 12px", borderRadius: 20, background: "#f5f5f4", color: "#57534e", fontSize: 12, fontWeight: 600 },

  top3Title: { fontSize: 12, color: "#a8a29e", textTransform: "uppercase", letterSpacing: 1, fontWeight: 600, marginBottom: 14, marginTop: 8 },
  barRow:    { display: "flex", alignItems: "center", gap: 10, marginBottom: 12 },
  barLabel:  { fontSize: 13, color: "#44403c", width: 180, flexShrink: 0 },
  barTrack:  { flex: 1, height: 8, background: "#f5f5f4", borderRadius: 4, overflow: "hidden" },
  barFill:   { height: "100%", borderRadius: 4, transition: "width .6s ease" },
  barVal:    { fontSize: 12, color: "#a8a29e", width: 38, textAlign: "right" },

  /* footer */
  footer: { textAlign: "center", padding: "24px", fontSize: 13, color: "#d6d3d1", borderTop: "1px solid #f5e6d8" },
};