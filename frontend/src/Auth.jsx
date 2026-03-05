import { useState } from "react";

const API = "http://127.0.0.1:8000";

export default function Auth({ onAuthenticated }) {
  const [mode, setMode]       = useState("login"); // "login" | "register"
  const [name, setName]       = useState("");
  const [email, setEmail]     = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [showPass, setShowPass] = useState(false);

  const reset = () => { setName(""); setEmail(""); setPassword(""); setError(null); };

  const switchMode = (m) => { setMode(m); reset(); };

  const handleSubmit = async () => {
    setError(null);
    if (!email || !password) { setError("Please fill in all fields."); return; }
    if (mode === "register" && !name) { setError("Please enter your name."); return; }

    setLoading(true);
    try {
      const body = mode === "login"
        ? { email, password }
        : { name, email, password };

      const res  = await fetch(`${API}/auth/${mode}`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Something went wrong");
      onAuthenticated(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => { if (e.key === "Enter") handleSubmit(); };

  return (
    <div style={s.page}>

      {/* ── animated background blobs ── */}
      <div style={{ ...s.blob, top: "5%",  left: "5%",  width: 380, height: 380, animationDelay: "0s"  }} />
      <div style={{ ...s.blob, top: "50%", right: "5%", width: 280, height: 280, animationDelay: "2s"  }} />
      <div style={{ ...s.blob, bottom: "5%", left: "30%", width: 200, height: 200, animationDelay: "4s" }} />

      {/* ── card ── */}
      <div style={s.card}>

        {/* logo */}
        <div style={s.logo}>
          <div style={s.logoDot} />
          <span style={s.logoText}>JobScan</span>
        </div>

        {/* headline */}
        <h1 style={s.title}>
          {mode === "login" ? (
            <>Welcome <span style={s.accent}>back</span></>
          ) : (
            <>Create your <span style={s.accent}>account</span></>
          )}
        </h1>
        <p style={s.subtitle}>
          {mode === "login"
            ? "Sign in to access your AI-powered job analysis"
            : "Join and discover your perfect career path"}
        </p>

        {/* tabs */}
        <div style={s.tabs}>
          <button style={{ ...s.tab, ...(mode === "login"    ? s.tabActive : {}) }} onClick={() => switchMode("login")}>Sign In</button>
          <button style={{ ...s.tab, ...(mode === "register" ? s.tabActive : {}) }} onClick={() => switchMode("register")}>Register</button>
        </div>

        {/* form */}
        <div style={s.form}>

          {mode === "register" && (
            <div style={s.field}>
              <label style={s.label}>Full Name</label>
              <div style={s.inputWrap}>
                <span style={s.inputIcon}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" stroke="#d4a57a" strokeWidth="2" strokeLinecap="round"/>
                    <circle cx="12" cy="7" r="4" stroke="#d4a57a" strokeWidth="2"/>
                  </svg>
                </span>
                <input
                  style={s.input}
                  type="text"
                  placeholder="Jane Dupont"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={handleKey}
                />
              </div>
            </div>
          )}

          <div style={s.field}>
            <label style={s.label}>Email Address</label>
            <div style={s.inputWrap}>
              <span style={s.inputIcon}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" stroke="#d4a57a" strokeWidth="2"/>
                  <polyline points="22,6 12,13 2,6" stroke="#d4a57a" strokeWidth="2"/>
                </svg>
              </span>
              <input
                style={s.input}
                type="email"
                placeholder="jane@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                onKeyDown={handleKey}
              />
            </div>
          </div>

          <div style={s.field}>
            <label style={s.label}>Password</label>
            <div style={s.inputWrap}>
              <span style={s.inputIcon}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2" stroke="#d4a57a" strokeWidth="2"/>
                  <path d="M7 11V7a5 5 0 0 1 10 0v4" stroke="#d4a57a" strokeWidth="2"/>
                </svg>
              </span>
              <input
                style={s.input}
                type={showPass ? "text" : "password"}
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onKeyDown={handleKey}
              />
              <button style={s.eyeBtn} onClick={() => setShowPass(!showPass)} tabIndex={-1}>
                {showPass ? (
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" stroke="#a8a29e" strokeWidth="2" strokeLinecap="round"/>
                    <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" stroke="#a8a29e" strokeWidth="2" strokeLinecap="round"/>
                    <line x1="1" y1="1" x2="23" y2="23" stroke="#a8a29e" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                ) : (
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" stroke="#a8a29e" strokeWidth="2"/>
                    <circle cx="12" cy="12" r="3" stroke="#a8a29e" strokeWidth="2"/>
                  </svg>
                )}
              </button>
            </div>
          </div>

          {/* error */}
          {error && (
            <div style={s.errorBox}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
                <circle cx="12" cy="12" r="10" stroke="#be123c" strokeWidth="2"/>
                <line x1="12" y1="8" x2="12" y2="12" stroke="#be123c" strokeWidth="2" strokeLinecap="round"/>
                <line x1="12" y1="16" x2="12.01" y2="16" stroke="#be123c" strokeWidth="2" strokeLinecap="round"/>
              </svg>
              {error}
            </div>
          )}

          {/* submit */}
          <button
            style={{ ...s.submitBtn, ...(loading ? s.submitBtnLoading : {}) }}
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? (
              <span style={s.loadingDots}>
                <span style={{ ...s.dot, animationDelay: "0s"   }} />
                <span style={{ ...s.dot, animationDelay: ".15s" }} />
                <span style={{ ...s.dot, animationDelay: ".3s"  }} />
              </span>
            ) : (
              mode === "login" ? "Sign In →" : "Create Account →"
            )}
          </button>
        </div>

        {/* footer switch */}
        <p style={s.switchText}>
          {mode === "login" ? "Don't have an account? " : "Already have an account? "}
          <button style={s.switchBtn} onClick={() => switchMode(mode === "login" ? "register" : "login")}>
            {mode === "login" ? "Register" : "Sign In"}
          </button>
        </p>
      </div>

      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px) scale(1); }
          50%       { transform: translateY(-24px) scale(1.05); }
        }
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: .4; }
          40%            { transform: translateY(-6px); opacity: 1; }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        input:focus { outline: none; }
        input::placeholder { color: #c4b5a5; }
        button:focus { outline: none; }
      `}</style>
    </div>
  );
}

/* ── STYLES ── */
const s = {
  page: {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#fffbf7",
    fontFamily: "'Segoe UI', sans-serif",
    position: "relative",
    overflow: "hidden",
    padding: "24px",
  },

  blob: {
    position: "absolute",
    borderRadius: "50%",
    background: "#ffedd5",
    filter: "blur(70px)",
    opacity: 0.55,
    animation: "float 7s ease-in-out infinite",
  },

  card: {
    position: "relative",
    zIndex: 1,
    width: "100%",
    maxWidth: 440,
    background: "#ffffff",
    border: "1px solid #f5e6d8",
    borderRadius: 24,
    padding: "40px 36px 32px",
    boxShadow: "0 20px 60px rgba(249,115,22,.10)",
  },

  logo:     { display: "flex", alignItems: "center", gap: 8, marginBottom: 28 },
  logoDot:  { width: 10, height: 10, borderRadius: "50%", background: "#f97316" },
  logoText: { fontWeight: 800, fontSize: 18, color: "#f97316" },

  title:    { fontSize: 30, fontWeight: 900, color: "#1c1917", lineHeight: 1.2, marginBottom: 8 },
  accent:   { color: "#f97316" },
  subtitle: { fontSize: 14, color: "#a8a29e", marginBottom: 28, lineHeight: 1.6 },

  tabs: {
    display: "flex",
    background: "#f5f5f4",
    borderRadius: 10,
    padding: 4,
    marginBottom: 28,
    gap: 4,
  },
  tab: {
    flex: 1,
    padding: "9px 0",
    borderRadius: 8,
    border: "none",
    background: "none",
    fontSize: 13,
    fontWeight: 600,
    color: "#a8a29e",
    cursor: "pointer",
    transition: "all .2s",
  },
  tabActive: {
    background: "#ffffff",
    color: "#1c1917",
    boxShadow: "0 1px 6px rgba(0,0,0,.08)",
  },

  form:  { display: "flex", flexDirection: "column", gap: 18 },
  field: { display: "flex", flexDirection: "column", gap: 6 },
  label: { fontSize: 12, fontWeight: 700, color: "#57534e", textTransform: "uppercase", letterSpacing: .6 },

  inputWrap: {
    display: "flex",
    alignItems: "center",
    background: "#fffbf7",
    border: "1.5px solid #f5e6d8",
    borderRadius: 10,
    overflow: "hidden",
    transition: "border-color .2s",
  },
  inputIcon: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: 42,
    flexShrink: 0,
  },
  input: {
    flex: 1,
    padding: "12px 12px 12px 0",
    border: "none",
    background: "transparent",
    fontSize: 14,
    color: "#1c1917",
    fontFamily: "inherit",
  },
  eyeBtn: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: 38,
    background: "none",
    border: "none",
    cursor: "pointer",
    flexShrink: 0,
  },

  errorBox: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "10px 14px",
    borderRadius: 10,
    background: "#fff1f2",
    border: "1px solid #fda4af",
    color: "#be123c",
    fontSize: 13,
    fontWeight: 500,
  },

  submitBtn: {
    width: "100%",
    padding: "14px",
    borderRadius: 12,
    background: "#1c1917",
    color: "#fff",
    fontWeight: 800,
    fontSize: 15,
    border: "none",
    cursor: "pointer",
    transition: "opacity .2s, transform .1s",
    marginTop: 4,
  },
  submitBtnLoading: { opacity: 0.6, cursor: "not-allowed" },

  loadingDots: { display: "flex", alignItems: "center", justifyContent: "center", gap: 5, height: 20 },
  dot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    background: "#fff",
    animation: "bounce 1s infinite",
  },

  switchText: { textAlign: "center", fontSize: 13, color: "#a8a29e", marginTop: 20 },
  switchBtn:  { background: "none", border: "none", color: "#f97316", fontWeight: 700, fontSize: 13, cursor: "pointer" },
};