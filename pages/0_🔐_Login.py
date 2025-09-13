# login.py
import streamlit as st
from time import sleep
from db import get_user, verify_password
import base64, os

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔐 Login",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="auto",
)

SHOW_BG_DEBUG = False  # True only when debugging paths

# ──────────────────────────────────────────────────────────────────────────────
# BG image helpers
# ──────────────────────────────────────────────────────────────────────────────
def _read_file_b64(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def _get_bg():
    for p in ["pages/log.jpg", "pages/cloud2.png", "static/cloud2.jpg", "static/log.png", "log.jpg", "log.png"]:
        if os.path.exists(p):
            return p, _read_file_b64(p)
    return None, None

bg_path, bg_b64 = _get_bg()

# ──────────────────────────────────────────────────────────────────────────────
# Navigation helpers
# ──────────────────────────────────────────────────────────────────────────────
def _go_to_register_button():
    try:
        st.switch_page("pages/1_Register.py")
    except Exception:
        for c in ["📝 Register", "Register", "1_Register", "pages/Register.py", "pages/register.py"]:
            try:
                st.switch_page(c); return
            except Exception:
                continue
        st.error("Could not find the Register page. Check that `pages/1_Register.py` exists and you run app from project root.")

def _go_to_weather() -> bool:
    """Admin landing"""
    candidates = [
        "weather_app.py",          # file at project root
        "pages/weather_app.py",    # file inside pages
        "🌦️ weather_app",          # page label with emoji
        "weather_app",             # page label plain
        "7-Day Weather Forecast",  # alt label
        "Home",                    # fallback
    ]
    for c in candidates:
        try:
            st.switch_page(c); return True
        except Exception:
            continue
    return False

def _go_to_user_dashboard() -> bool:
    """Registered user landing"""
    candidates = [
        "pages/3_User_Dashboard.py",
        "3_User_Dashboard.py",
        "🧭 User Dashboard",
        "User Dashboard",
        "3_User_Dashboard",
        "pages/User_Dashboard.py",
    ]
    for c in candidates:
        try:
            st.switch_page(c); return True
        except Exception:
            continue
    return False

# ──────────────────────────────────────────────────────────────────────────────
# RBAC helper
# ──────────────────────────────────────────────────────────────────────────────
def _is_admin(u: str, p: str) -> bool:
    return u.strip().lower() == "kamal1" and p == "hello123kamal"

# ──────────────────────────────────────────────────────────────────────────────
# CSS — look & feel
# ──────────────────────────────────────────────────────────────────────────────
if bg_b64:
    bg_css = f"""
    html, body, .stApp, [data-testid="stAppViewContainer"], section.main {{
      background: lightblue;
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
      --topbar: #0a1c3d;
      --topbar-border: rgba(255,255,255,.08);
      --brand-blue: #0a1c3d;
    }}

    {bg_css}

    header[data-testid="stHeader"] {{
      background: #0a1c3d !important;
      border-bottom: 1px solid rgba(255,255,255,.08) !important;
    }}
    header[data-testid="stHeader"] * {{ color: #e6f0ff !important; }}

    .block-container {{
      padding-top: 0 !important; padding-bottom: 0 !important;
      min-height: 100dvh; display:flex; align-items:center; justify-content:center; max-width: 820px;
      position: relative; z-index: 1;
    }}

    .login-card {{
      width: min(800px, 96vw);
      padding: 28px 26px 24px 26px; border-radius: 22px;
      background: rgba(0,0,0,0.55);
      border: 1px solid rgba(148,163,184,0.28);
      backdrop-filter: blur(16px);
      box-shadow: 0 24px 60px rgba(2,6,23,.18);
      color: #fff;
    }}

    .submini {{ color:black; font-size:15px; letter-spacing:.10em; font-weight:800; }}
    .heading {{ font-weight:900; font-size:30px; margin:6px 0 2px; letter-spacing:.01em; }}
    .hint {{ color:black; font-size:15px; margin:0 0 14px; }}

    .stTextInput>div>div>input {{ padding-top: 12px; padding-bottom: 12px; font-weight:600; }}
    .stTextInput label {{ font-weight: 700; }}
    .stCheckbox>label {{ font-weight: 600; }}

    .stTextInput input:focus, .stCheckbox input:focus, .stToggle input:focus {{
      outline: none !important; box-shadow: 0 0 0 4px rgba(59,130,246,.25) !important; border-radius: 12px;
    }}

    .stButton>button, button[kind="primary"] {{
      height: 48px; border-radius: 999px !important; font-weight: 900; letter-spacing:.02em;
      background: #b1e0e0; border: none; color: black !important;
      box-shadow: 0 10px 18px rgba(2,6,23,.15);
      width: auto !important; padding: 0 40px;
    }}
    .stButton>button:hover {{ filter: none !important; }}
    .stButton>button:active {{ transform: translateY(1px); }}

    /* Inline sign-up row */
    .signup-row {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 10px;
      margin-left: 40px;
      color: black;
    }}
    .signup-row .pill-btn {{
      display:inline-flex; align-items:center; justify-content:center;
      height: 44px; padding: 0 22px; border-radius: 999px;
      background: gray; font-weight: 900; letter-spacing:.02em;
      text-decoration: none; color: #000;
      box-shadow: 0 10px 18px rgba(2,6,23,.15);
    }}
    .signup-row .pill-btn:active {{ transform: translateY(1px); }}
    
    </style>
    """,
    unsafe_allow_html=True,
)
# Sidebar look
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F3D7A 0%, #0A2B5E 50%, #062048 100%) !important;
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="login-card">', unsafe_allow_html=True)
st.markdown('<div class="submini">WELCOME BACK</div>', unsafe_allow_html=True)
st.markdown('<div class="heading">Sign in to continue 🔐</div>', unsafe_allow_html=True)
st.markdown('<div class="hint">Use your username or email and password to access the dashboard.</div>', unsafe_allow_html=True)

with st.form("login_form", clear_on_submit=False, enter_to_submit=True):
    c1, c2 = st.columns([10, 1])
    with c1:
        username = st.text_input("Email / Username", placeholder="👤 e.g., admin", help="Usernames are case-insensitive")
    with c2:
        show_pw = st.toggle("", value=False, key="show_pw")

    password = st.text_input(
        "Password",
        type=("default" if show_pw else "password"),
        placeholder="🔒 ••••••••",
        help="Keep your account secure",
    )
    remember = st.checkbox("Remember me", value=True, help="Stay signed in on this device")

    left, mid, right = st.columns([1, 1, 1])
    with mid:
        login_btn = st.form_submit_button("Log in to the Dashboard")

# —— Inline "Don't have an account?  [Create one]"
signup_col1, signup_col2 = st.columns([1, 2])
with signup_col1:
    st.markdown('<div class="signup-row"><span>Don’t have an account?</span></div>', unsafe_allow_html=True)
with signup_col2:
    if hasattr(st, "page_link"):
        st.page_link("pages/1_Register.py", label="Create new account", icon="➕")
    else:
        if st.button("Create one", key="create_one_fallback"):
            _go_to_register_button()
            st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Submit handler with RBAC
# ──────────────────────────────────────────────────────────────────────────────
if login_btn:
    u = (username or "").strip()
    p = (password or "").strip()

    if not u or not p:
        st.error("Please enter **both** username and password.")
    elif len(p) < 4:
        st.error("Password seems too short. Please check and try again.")
    else:
        # 1) Admin hard-gate (doesn't need to be in DB)
        if _is_admin(u, p):
            st.session_state.update({
                "auth_ok": True,
                "auth_user": "kamal1",
                "auth_name": "Administrator",
                "auth_role": "admin",
                "auth_remember": bool(remember),
            })
            st.success("Admin login successful ✅ Redirecting…")
            sleep(0.05)
            if not _go_to_weather():
                st.warning("Could not navigate to weather_app.py. Check the file name/location.")
                st.rerun()
            st.stop()

        # 2) Normal registered users (must exist in auth.db)
        row = get_user(u)  # expected: (id, username, full_name, password_hash)
        if row and verify_password(p, row[3]):
            st.session_state.update({
                "auth_ok": True,
                "auth_user": row[1],
                "auth_name": row[2],
                "auth_role": "user",
                "auth_remember": bool(remember),
            })
            st.success("Login successful ✅ Redirecting…")
            sleep(0.05)
            if not _go_to_user_dashboard():
                st.warning("Could not navigate to 3_User_Dashboard page. Check the file name/location.")
                st.rerun()
            st.stop()
        else:
            st.error("Invalid username or password")

st.markdown('</div>', unsafe_allow_html=True)  # close .login-card
