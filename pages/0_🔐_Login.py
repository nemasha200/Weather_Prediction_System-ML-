# pages/0_Login.py
import streamlit as st
from time import sleep
from db import get_user, verify_password
import base64, os

st.set_page_config(
    page_title="🔐 Login",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SHOW_BG_DEBUG = False  # True only when debugging paths

# ---------------- BG helpers ----------------
def _read_file_b64(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def _get_bg():
    for p in ["pages/log.jpg", "pages/log.png", "static/log.jpg", "static/log.png", "log.jpg", "log.png"]:
        if os.path.exists(p):
            return p, _read_file_b64(p)
    return None, None

bg_path, bg_b64 = _get_bg()
if SHOW_BG_DEBUG:
    dbg = st.empty()
    if bg_b64:
        dbg.caption(f"✅ Background loaded from **{bg_path}**")
    else:
        dbg.warning("⚠️ Could not find log.jpg/log.png. Using fallback.")

# ---------------- Query param helpers ----------------
try:
    _HAS_QP = hasattr(st, "query_params")
except Exception:
    _HAS_QP = False

def _get_qp_dict():
    if _HAS_QP:
        try:
            return dict(st.query_params)
        except Exception:
            pass
    try:
        return st.experimental_get_query_params()
    except Exception:
        return {}

def _set_qp_dict(d: dict):
    if _HAS_QP:
        try:
            st.query_params = d
            return
        except Exception:
            pass
    try:
        st.experimental_set_query_params(**d)
    except Exception:
        pass

# ---------------- Handle ?logout ----------------
qp = _get_qp_dict()
if any(k.lower() == "logout" for k in qp.keys()):
    st.session_state["auth_ok"] = False
    st.session_state.pop("auth_user", None)
    st.session_state.pop("auth_name", None)
    st.session_state["auth_remember"] = False
    _set_qp_dict({k: v for k, v in qp.items() if k.lower() != "logout"})
    st.toast("Logged out", icon="✅")

# ---------------- CSS ----------------
if bg_b64:
    bg_css = f"""
    html, body, .stApp, [data-testid="stAppViewContainer"], section.main {{
      background: transparent !important;
    }}
    body::before {{
      content: "";
      position: fixed; inset: 0; z-index: 0; pointer-events: none;
      background: url("data:image/jpg;base64,{bg_b64}") no-repeat center center fixed;
      background-size: cover;
    }}
    """
else:
    bg_css = """
    html, body, .stApp, [data-testid="stAppViewContainer"], section.main { background: transparent !important; }
    body::before { content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
      background: linear-gradient(135deg, #0b1220 0%, #0f1a14 100%); }
    """

st.markdown(
    f"""
    <style>
    :root {{
      --topbar: #0a1c3d;           /* dark blue you want */
      --topbar-border: rgba(255,255,255,.08);
    }}

    /* Hide sidebar + footer chrome but keep toolbar visible */
    div[data-testid="stSidebar"], div[data-testid="stSidebarNav"] {{ display: none !important; }}
    footer {{ visibility: hidden !important; }}

    {bg_css}

    /* ===== TOP HEADER BAR ONLY (dark blue) ===== */
    /* Works across Streamlit versions using data-testid */
    header[data-testid="stHeader"] {{
      background: var(--topbar) !important;
      border-bottom: 1px solid var(--topbar-border) !important;
    }}
    header[data-testid="stHeader"] > div {{      /* inner wrapper */
      background: var(--topbar) !important;
    }}
    /* Fallback for specific emotion class seen in your app */
    .st-emotion-cache-1ffuo7c {{
      background: var(--topbar) !important;
    }}
    /* Make text/icons readable on dark blue */
    header[data-testid="stHeader"] * {{
      color: #e6f0ff !important;
    }}

    /* Center container above background layer */
    .block-container {{
      padding-top: 0 !important; padding-bottom: 0 !important;
      min-height: 100dvh; display:flex; align-items:center; justify-content:center; max-width: 820px;
      position: relative; z-index: 1;
    }}

    /* Glass card */
    .login-card {{
      width: min(800px, 96vw);
      padding: 28px 26px 24px 26px; border-radius: 22px;
      background: rgba(0,0,0,0.55);
      border: 1px solid rgba(148,163,184,0.28);
      backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
      box-shadow: 0 24px 60px rgba(2,6,23,.18);
      transition: transform .2s ease;
      color: #fff;
    }}
    .login-card:hover {{ transform: translateY(-1px); }}

    .submini {{ color:#cbd5e1; font-size:.85rem; letter-spacing:.10em; font-weight:800; }}
    .heading {{ font-weight:900; font-size:28px; margin:6px 0 2px; letter-spacing:.01em; }}
    .hint {{ color:#d1d5db; font-size:.92rem; margin:0 0 14px; }}

    section.main div[data-baseweb="input"] > div {{ border-radius: 14px; }}
    .stTextInput>div>div>input {{ padding-top: 12px; padding-bottom: 12px; font-weight:600; }}
    .stTextInput label {{ font-weight: 700; }}
    .stToggle > label {{ font-weight: 600; }}
    .stCheckbox>label {{ font-weight: 600; }}

    .stTextInput input:focus, .stCheckbox input:focus, .stToggle input:focus {{
      outline: none !important; box-shadow: 0 0 0 4px rgba(59,130,246,.25) !important; border-radius: 12px;
    }}

    .stButton>button, button[kind="primary"] {{
      height: 48px; border-radius: 999px !important; font-weight: 900; letter-spacing:.02em;
      background: linear-gradient(90deg, #1e40af 0%, #0ea5e9 100%) !important;
      border: none; color: #fff !important; width: 100%;
      box-shadow: 0 10px 18px rgba(2,6,23,.15); transition: transform .06s ease, filter .12s ease;
    }}
    .stButton>button:hover {{ filter: brightness(1.03); }}
    .stButton>button:active {{ transform: translateY(1px); }}

    .actions-row {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap: 10px; margin-top: 8px; }}
    .link-btn>button {{
      background: transparent !important; color:#93c5fd !important; box-shadow:none !important;
      border: none !important; height: auto; padding: 0 !important; font-weight:800; text-decoration: underline;
    }}
    .link-btn>button:hover {{ opacity:.9 }}

    .top-links {{ position:fixed; top:12px; right:18px; z-index:10000; display:flex; gap:10px; }}
    .top-links a {{
      background:#ffffffcc; color:#003366; font-weight:800; text-decoration:none; padding:8px 14px;
      border-radius:999px; border:1px solid #e5e7eb; box-shadow:0 2px 6px rgba(0,0,0,.08); backdrop-filter:blur(8px);
    }}
    [data-theme="dark"] .top-links a {{ background:#0b1220cc; color:#e6f0ff; border-color:#1f2937; }}
    .top-links a:hover {{ filter:brightness(1.05); }}

    .tiny {{ font-size:.84rem; color:#e5e7eb; }}
    </style>
    <div class="top-links"><a class="logout" href="?logout=1">Logout</a></div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Redirect helper ----------------
def _go_to_weather() -> bool:
    for t in ["7-Day Weather Forecast", "🌦️ weather_app", "weather_app.py", "Home"]:
        try:
            st.switch_page(t); return True
        except Exception:
            continue
    return False

# ---------------- Auto-redirect ----------------
if st.session_state.get("auth_ok") and st.session_state.get("auth_remember", True):
    if _go_to_weather(): st.stop()
    else: st.info("You're already signed in. Open the main page from the sidebar.", icon="✅")

# ---------------- UI ----------------
st.markdown('<div class="login-card">', unsafe_allow_html=True)
st.markdown('<div class="submini">WELCOME BACK</div>', unsafe_allow_html=True)
st.markdown('<div class="heading">Sign in to continue 🔐</div>', unsafe_allow_html=True)
st.markdown('<div class="hint">Use your username or email and password to access the dashboard.</div>', unsafe_allow_html=True)

with st.form("login_form", clear_on_submit=False, enter_to_submit=True):
    c1, c2 = st.columns([1, 1])
    with c1:
        username = st.text_input("Email / Username", placeholder="👤 e.g., admin", help="Usernames are case-insensitive")
    with c2:
        show_pw = st.toggle("Show password", value=False, key="show_pw")

    password = st.text_input("Password", type=("default" if show_pw else "password"),
                             placeholder="🔒 ••••••••", help="Keep your account secure")
    remember = st.checkbox("Remember me", value=True, help="Stay signed in on this device")
    login_btn = st.form_submit_button("Log in")

# ---------------- Submit handler ----------------
if login_btn:
    u = (username or "").strip()
    p = (password or "").strip()
    if not u or not p:
        st.error("Please enter **both** username and password.")
    elif len(p) < 4:
        st.error("Password seems too short. Please check and try again.")
    else:
        row = get_user(u)  # (id, username, full_name, password_hash)
        if row and verify_password(p, row[3]):
            st.session_state.update({
                "auth_ok": True,
                "auth_user": row[1],
                "auth_name": row[2],
                "auth_remember": bool(remember),
            })
            if SHOW_BG_DEBUG: dbg.empty()
            st.success("Login successful ✅ Redirecting…")
            sleep(0.05)
            _go_to_weather()
            st.stop()
        else:
            st.error("Invalid username or password")

# ---------------- Bottom actions ----------------
st.markdown('<div class="actions-row">', unsafe_allow_html=True)
st.caption("🔁 Forgot password? ", help="Ask your admin to reset your password.")
st.markdown('<div class="link-btn">', unsafe_allow_html=True)
if st.button("Create a new account"):
    try:
        st.switch_page("pages/1_Register.py")
    except Exception:
        try:
            st.switch_page("📝 Register")
        except Exception:
            qp = _get_qp_dict(); qp.update({"register": "1"}); _set_qp_dict(qp); st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
