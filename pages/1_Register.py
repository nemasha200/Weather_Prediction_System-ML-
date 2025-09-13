# pages/1_Register.py — Create account page (uses db.create_user)
import streamlit as st
from db import create_user, get_user
import base64, os

st.set_page_config(
    page_title="📝 Register",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SHOW_BG_DEBUG = False  # set True only when debugging bg paths

# ======================
# BG image helpers
# ======================
def _read_file_b64(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def _get_bg():
    for p in ["pages/log.jpg", "pages/cloud2.png", "static/cloud2.jpg",
              "static/log.png", "log.jpg", "log.png"]:
        if os.path.exists(p):
            return p, _read_file_b64(p)
    return None, None

bg_path, bg_b64 = _get_bg()

# ======================
# Query param helpers
# ======================
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


# ======================
# CSS styling
# ======================
if bg_b64:
    bg_css = f"""
    html, body, .stApp, [data-testid="stAppViewContainer"], section.main {{
      background: lightblue;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      z-index: 0;
      pointer-events: none;
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
}}

{bg_css}

header[data-testid="stHeader"] {{
  background: var(--topbar) !important;
  border-bottom: 1px solid var(--topbar-border) !important;
}}
header[data-testid="stHeader"] * {{
  color: #e6f0ff !important;
}}

.card{{
  width: min(560px, 96vw);
  background: gray;
  border: 1px solid rgba(148,163,184,0.28);
  border-radius: 22px;
  padding: 28px 26px 24px;
  box-shadow: 0 24px 60px rgba(2,6,23,.18);
  backdrop-filter: blur(16px);
}}

.heading{{ font-weight:900; font-size:30px; margin: 6px 0 4px 0; }}
.subnote{{ color:black; margin: 0 0 14px 0; font-size:15px; }}

div.stButton > button {{
  height: 48px;
  width: auto;
  padding: 0 40px;
  background: #0a1c3d !important;
  color: #fff !important;
  border-radius: 999px !important;
  border: none !important;
  font-weight: 900 !important;
  font-size: 1.1rem;
  box-shadow: 0 6px 16px rgba(0,0,0,.4);
}}
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

# --------------------
# Form card
# --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="heading">Create account 📝</div>', unsafe_allow_html=True)
st.markdown('<div class="subnote">Fill the details below to get started.</div>', unsafe_allow_html=True)

with st.form("register_form", clear_on_submit=False):
    full_name = st.text_input("Full name")
    username  = st.text_input("Username (or email)")
    pw        = st.text_input("Password", type="password")
    pw2       = st.text_input("Confirm password", type="password")
    agree     = st.checkbox("I agree to the terms")

    # Put button in rightmost column
    left, mid, right = st.columns([2, 1, 1])
    with right:
        submitted = st.form_submit_button("Create account")

    if submitted:
        u  = (username or "").strip()
        p1 = (pw or "").strip()
        p2 = (pw2 or "").strip()

        if not full_name or not u or not p1 or not p2:
            st.error("Please fill all fields.")
        elif len(p1) < 6:
            st.error("Password must be at least 6 characters.")
        elif p1 != p2:
            st.error("Passwords do not match.")
        elif not agree:
            st.error("Please agree to the terms.")
        else:
            try:
                if get_user(u):
                    st.error("This username already exists.")
                else:
                    create_user(username=u, full_name=full_name, password=p1, scheme="sha256")
            except Exception as e:
                st.error(f"Could not create account: {e}")
            else:
                st.success("✅ Account created! Redirecting to Login…")
                for target in ("pages/0_🔐_Login.py", "pages/0_Login.py", "🔐 Login"):
                    try:
                        st.switch_page(target)
                        break
                    except Exception:
                        pass
                else:
                    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# --------------------
# Back to Login button
# --------------------
st.write("")
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    if st.button("Back to Login"):
        for target in ("pages/0_🔐_Login.py", "pages/0_Login.py", "🔐 Login"):
            try:
                st.switch_page(target)
                break
            except Exception:
                pass
        else:
            qp = _get_qp_dict()
            qp.update({"login": "1"})
            _set_qp_dict(qp)
            st.rerun()
