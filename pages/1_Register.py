# pages/1_Register.py — Create account page (uses db.create_user)
import streamlit as st
from db import create_user, get_user

st.set_page_config(
    page_title="📝 Register",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------
# Styles (modern UI only — logic unchanged)
# --------------------
st.markdown(
    """
<style>
/* Hide app chrome for a clean auth screen */
div[data-testid="stSidebar"], div[data-testid="stSidebarNav"] { display: none !important; }
header [data-testid="stToolbar"] { visibility: hidden !important; }
footer { visibility: hidden !important; }

/* Background: soft gradients + dark-mode aware */
html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 600px at 20% 10%, rgba(59,130,246,.10), transparent 50%),
    radial-gradient(1200px 600px at 80% 90%, rgba(16,185,129,.10), transparent 50%),
    linear-gradient(135deg, #eef2ff 0%, #f0fdf4 100%) !important;
}
[data-theme="dark"] html,
[data-theme="dark"] body,
[data-theme="dark"] [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 600px at 20% 10%, rgba(59,130,246,.20), transparent 50%),
    radial-gradient(1200px 600px at 80% 90%, rgba(16,185,129,.18), transparent 50%),
    linear-gradient(135deg, #0b1220 0%, #0f1a14 100%) !important;
}

/* Center content nicely */
.block-container{
  padding-top: 0 !important; padding-bottom: 0 !important;
  min-height: 100dvh; display:flex; align-items:center; justify-content:center; max-width: 820px;
}

/* Card: glassy with subtle lift on hover */
.card{
  width: min(560px, 96vw);
  background: rgba(255,255,255,0.85) !important;
  border: 1px solid rgba(148,163,184,0.28);
  border-radius: 22px;
  padding: 28px 26px 24px;
  box-shadow: 0 24px 60px rgba(2,6,23,.18);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  transition: transform .18s ease, box-shadow .18s ease;
}
.card:hover{ transform: translateY(-1px); box-shadow: 0 28px 70px rgba(2,6,23,.22); }
[data-theme="dark"] .card{
  background: rgba(2,6,23,0.55) !important;
  border-color: rgba(148,163,184,0.22);
}

/* Title + tiny helper */
.heading{ font-weight:900; font-size:28px; margin: 6px 0 4px 0; letter-spacing:.01em; }
.subnote{ color:#64748b; margin: 0 0 14px 0; font-size:.94rem; }
[data-theme="dark"] .subnote{ color:#94a3b8; }

/* Inputs: rounded, thicker padding, consistent background */
.card [data-baseweb="input"] > div { border-radius: 14px; }
.card .stTextInput>div>div>input,
.card .stTextArea textarea {
  background:#9ec2c45c !important;
  color:#0f172a !important;
  padding-top: 12px; padding-bottom: 12px;
  font-weight:600;
  border:1px solid rgba(0,0,0,.15) !important;
  border-radius:12px !important;
}
[data-theme="dark"] .card .stTextInput>div>div>input,
[data-theme="dark"] .card .stTextArea textarea {
  background:#0b1220 !important; color:#e5e7eb !important; border-color:#1f2937 !important;
}

/* Focus ring: emerald glow */
.card input:focus, .card textarea:focus {
  outline:none !important;
  box-shadow:0 0 0 4px rgba(16,185,129,.30) !important;
  border-color:#10b981 !important;
}

/* Labels & checkbox text */
.card label, .card .stCheckbox { color: inherit !important; font-weight:700; }

/* Primary button (Create account): bold pill with micro-interaction */
div.stButton > button {
  height: 48px; width: 100%;
  background: linear-gradient(90deg, #059669 0%, #10b981 100%) !important;
  color:#fff !important; border-radius:999px !important; border:none !important;
  font-weight:900 !important; letter-spacing:.02em;
  box-shadow: 0 10px 18px rgba(2,6,23,.15);
  transition: transform .06s ease, filter .12s ease;
}
div.stButton > button:hover { filter:brightness(1.04); }
div.stButton > button:active { transform: translateY(1px); }

/* Back to Login (centered) */
.actions-row{ display:flex; justify-content:center; margin-top: 16px; }
.actions-row .back-btn > button{
  background:#10b981 !important; color:#fff !important; border:none !important;
  border-radius:999px !important; padding:10px 22px !important; font-weight:900 !important;
  box-shadow: 0 6px 14px rgba(2,6,23,.12);
}

/* Small helper text under inputs if needed */
.tiny{ font-size:.85rem; color:#6b7280; }
[data-theme="dark"] .tiny{ color:#9ca3af; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------
# Form card (markup unchanged except a subnote line)
# --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="heading">Create account</div>', unsafe_allow_html=True)
st.markdown('<div class="subnote">Fill the details below to get started.</div>', unsafe_allow_html=True)

with st.form("register_form", clear_on_submit=False):
    full_name = st.text_input("Full name")
    username  = st.text_input("Username (or email)")
    pw        = st.text_input("Password", type="password")
    pw2       = st.text_input("Confirm password", type="password")
    agree     = st.checkbox("I agree to the terms")
    submitted = st.form_submit_button("Create account")

    # ---------- Submit handler (UNCHANGED) ----------
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
                    # Note: create_user expects password=...
                    create_user(username=u, full_name=full_name, password=p1, scheme="sha256")
            except Exception as e:
                st.error(f"Could not create account: {e}")
            else:
                st.success("✅ Account created! Redirecting to Login…")
                # Try multiple targets to be robust to filename/label
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
# Back to Login (button + navigation — UNCHANGED behavior)
# --------------------
st.write("")  # small spacing
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    st.markdown('<div class="actions-row"><div class="back-btn">', unsafe_allow_html=True)
    if st.button("Back to Login"):
        for target in ("pages/0_🔐_Login.py", "pages/0_Login.py", "🔐 Login"):
            try:
                st.switch_page(target)
                break
            except Exception:
                pass
        else:
            st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)
