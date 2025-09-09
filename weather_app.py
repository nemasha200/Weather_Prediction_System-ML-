# weather_app.py  — top of file (routing + auth)
import os, io, math, base64, numpy as np, pandas as pd
from datetime import datetime, timedelta
import streamlit as st, plotly.express as px, requests
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="7-Day Weather Forecast", layout="wide", page_icon="🌦️")

# ---------- NAV HANDLERS (run BEFORE any st.stop) ----------
_qp = dict(st.query_params)

# ---------- NAV HANDLERS (run BEFORE any st.stop) ----------
def _qp_get():
    try:
        return dict(st.query_params)                      # Streamlit ≥1.30
    except Exception:
        return {k: (v[0] if isinstance(v, list) and len(v)==1 else v)
                for k, v in st.experimental_get_query_params().items()}  # older

def _qp_set(d: dict):
    try:
        st.query_params = d                              # Streamlit ≥1.30
    except Exception:
        st.experimental_set_query_params(**d)            # older

_qp = {k.lower(): v for k, v in _qp_get().items()}


# Logout (works even if not logged in)
if any(k.lower() == "logout" for k in _qp):
    st.session_state.clear()
    st.query_params = {k: v for k, v in _qp.items() if k.lower() != "logout"}
    try:
        st.switch_page("pages/0_Login.py")
    except Exception:
        try:
            st.switch_page("🔐 Login")
        except Exception:
            st.rerun()

# Jump to Users admin (deep-link + auth-aware)
if any(k.lower() == "users" for k in _qp):
    # remove the query param so we don't loop
    st.query_params = {k: v for k, v in _qp.items() if k.lower() != "users"}

    if st.session_state.get("auth_ok", False):
        # already logged in → go straight to Users
        for target in ("pages/2_Users.py", "👥 Users", "pages/2_People.py", "👥 People", "people.py"):
            try:
                st.switch_page(target)
                st.stop()            # prevent the rest of this page from running
            except Exception:
                continue
        st.rerun()
    else:
        # not logged in → remember intent and go to Login
        st.session_state["after_login"] = "users"
        for target in ("pages/0_🔐_Login.py", "pages/0_Login.py", "🔐 Login"):
            try:
                st.switch_page(target)
                st.stop()
            except Exception:
                continue
        st.rerun()


# Jump to Register
if any(k.lower() == "register" for k in _qp):
    st.query_params = {k: v for k, v in _qp.items() if k.lower() != "register"}
    for target in ("pages/1_Register.py", "📝 Register"):
        try:
            st.switch_page(target)
            break
        except Exception:
            continue
    else:
        st.rerun()

    # ---------- AUTH GUARD (redirect to Login if not logged in) ----------
if not st.session_state.get("auth_ok", False):
    # go straight to Login page
    for target in ("pages/0_🔐_Login.py", "pages/0_Login.py", "🔐 Login"):
        try:
            st.switch_page(target)
            st.stop()   # stop executing this page
        except Exception:
            pass
    # final fallback if switch_page isn't available
    st.warning("Please log in first (see 🔐 Login page).", icon="🔑")
    st.stop()

    
    # ─────────── Admin Guard (put near the top of weather_app.py) ───────────
import streamlit as st

# Not logged in? → go to Login
if not st.session_state.get("auth_ok", False):
    for target in ("pages/0_🔐_Login.py", "pages/0_Login.py", "🔐 Login", "login.py"):
        try:
            st.switch_page(target); st.stop()
        except Exception:
            pass
    st.error("Please log in first."); st.stop()

# Logged in but not admin? → bounce to User Dashboard
if st.session_state.get("auth_role") != "admin":
    st.warning("Admins only. Redirecting to your dashboard…", icon="🔒")
    for target in ("pages/3_User_Dashboard.py", "🧭 User Dashboard", "User Dashboard"):
        try:
            st.switch_page(target); st.stop()
        except Exception:
            pass
    st.stop()
# ────────────────────────────────────────────────────────────────────────




# ----------------------
# Background image (local file)
# ----------------------
def _b64_image(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

_bg64 = _b64_image(r"C:\Users\Nemasha\Desktop\five_models\LSTM1\cloud2.jpg")
if _bg64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{_bg64}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Background image not found.", icon="⚠️")

# Button colors (your next style block)
st.markdown("""<style> ... </style>""", unsafe_allow_html=True)


# Top header bar
st.markdown("""
<style>
header[data-testid="stHeader"] {
    background-color: #003366 !important;
    height: 60px;
    display: flex; justify-content: center; align-items: center; position: relative;
}
header[data-testid="stHeader"]::before {
    content: "Weather Prediction System";
    font-size: 30px; font-weight: bold; color: white;
    position: absolute; left: 50%; transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

# Fixed top-right action buttons
st.markdown("""
<style>
.logout-btn{
  position:fixed; top:72px; right:18px; z-index:2147483647; display:flex; gap:8px;
}
.logout-btn a{
  color:#ffffff !important; font-weight:700; text-decoration:none !important;
  padding:8px 14px; border-radius:8px; border:1px solid rgba(255,255,255,.25) !important;
  box-shadow:0 2px 6px rgba(0,0,0,.12); display:inline-block;
}
.logout-btn a.users{ background:#2563eb !important; }
.logout-btn a.users:hover{ background:#1d4ed8 !important; }
.logout-btn a.logout{ background:#dc2626 !important; }
.logout-btn a.logout:hover{ background:#b91c1c !important; }
@media (max-width:640px){ .logout-btn{ top:64px; right:12px; } }
</style>
<div class="logout-btn">
  <a class="users"    href="?users=1">See users</a>
  <a class="logout"   href="?logout=1">Logout</a>
</div>


""", unsafe_allow_html=True)

# Button colors
st.markdown("""
<style>
div.stButton > button {
    background-color: #54a8a5 !important; color: white !important; border: none;
    font-weight: bold; border-radius: 8px; height: 3em; transition: 0.3s;
}
div.stButton > button:hover { background-color: #008000 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar look
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F3D7A 0%, #0A2B5E 50%, #062048 100%) !important;
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# Card/Badge styles
st.markdown("""
<style>
:root{
  --card-bg: rgba(255,255,255,0.62);
  --card-brd: rgba(255,255,255,0.35);
  --shadow: 0 10px 30px rgba(2,6,23,0.10);
}
@media (prefers-color-scheme: dark){
  :root{
    --card-bg: rgba(15,23,42,0.55);
    --card-brd: rgba(148,163,184,0.20);
    --shadow: 0 10px 30px rgba(0,0,0,0.35);
  }
}
.card{background:var(--card-bg);border:1px solid var(--card-brd);border-radius:16px;padding:16px 18px;box-shadow:var(--shadow);backdrop-filter:blur(10px)}
.badge{display:inline-flex;align-items:center;gap:.5rem;padding:.25rem .6rem;border-radius:999px;border:1px solid var(--card-brd);font-size:.85rem;opacity:.9}
.kpi{display:flex;flex-direction:column;gap:.25rem}
.kpi .label{opacity:.75;font-size:.85rem}
.kpi .value{font-size:1.6rem;font-weight:700;line-height:1.1}
.header{border-radius:18px;padding:18px 20px;background:linear-gradient(135deg, rgba(59,130,246,.15), rgba(16,185,129,.15));border:1px solid var(--card-brd);box-shadow:var(--shadow)}
hr.sep{border:none;border-top:1px solid var(--card-brd);margin:8px 0 0 0}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{padding:10px 12px}
.daycard{display:flex;align-items:center;justify-content:space-between;border:1px dashed var(--card-brd);border-radius:14px;padding:10px 12px;margin-bottom:8px}
.daymeta{opacity:.85}
.progress{height:10px;border-radius:999px;background:#e5e7eb;}
.bar{height:10px;border-radius:999px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.yellow-card{
  background: #FFF8DB;              /* soft yellow */
  border: 1px solid #F6E27D;        /* golden border */
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 10px 30px rgba(2,6,23,0.08);
}
.yellow-card h4, .yellow-card h3, .yellow-card h2, .yellow-card h1{
  color: #5B4E00;                   /* darker text for headings */
}
.yellow-badge{
  display:inline-flex;align-items:center;gap:.5rem;
  padding:.25rem .6rem;border-radius:999px;
  background:#FFF0A6;border:1px solid #F1D775;opacity:.95;
}
.yellow-sep{
  border:none;border-top:1px dashed #EAD26E;margin:8px 0 0 0;
}
</style>
""", unsafe_allow_html=True)


# Header row
with st.container():
    colA, colB = st.columns([1, .15], vertical_alignment="center")
    with colA:
        st.markdown('<div class="header">', unsafe_allow_html=True)
        st.markdown("### 🌦️ 7-Day Weather Forecast")
        st.markdown(
            '<div class="subtle">Enter today’s weather or upload the last 30 days — or fetch today automatically by city. Your LSTM model predicts the next 7 days.</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown('<span class="badge">🧠 LSTM</span><br>', unsafe_allow_html=True)
        st.caption("Sequence: 30 • Horizon: 7")
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------
# Constants & helpers
# ----------------------
FEATURE_COLUMNS = [
    "temp", "humidity", "precip", "windspeed", "winddir",
    "cloudcover", "dew", "uvindex", "sealevelpressure"
]
ALL_FEATURES = FEATURE_COLUMNS + [
    "day_of_year", "month", "day_of_week",
    "day_sin", "day_cos", "month_sin", "month_cos"
]

@st.cache_resource
def load_model_and_scaler():
    model_path_h5 = os.path.join("lstm", "lstm_model.h5")
    keras_path = os.path.join("lstm", "lstm_model.keras")
    scaler_path = os.path.join("lstm", "scaler.joblib")

    if not os.path.exists(scaler_path):
        alt_scaler = "scaler.joblib"
        if os.path.exists(alt_scaler):
            scaler_path = alt_scaler
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path} or {alt_scaler}")

    if os.path.exists(model_path_h5):
        model_path = model_path_h5
    elif os.path.exists(keras_path):
        model_path = keras_path
    elif os.path.exists("lstm_model.h5"):
        model_path = "lstm_model.h5"
    elif os.path.exists("lstm_model.keras"):
        model_path = "lstm_model.keras"
    else:
        raise FileNotFoundError("Model not found (looked in ./lstm and current folder).")

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler, model_path, scaler_path

def add_date_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for c in FEATURE_COLUMNS:
        if c in df.columns:
            df[c] = df[c].fillna(method="ffill").fillna(method="bfill")
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df

def build_today_row(temp, humidity, precip, windspeed, winddir, cloudcover, dew, uvindex, sealevelpressure, now=None):
    now = now or datetime.now()
    doy = now.timetuple().tm_yday
    month = now.month
    dow = now.weekday()
    day_sin = np.sin(2 * np.pi * doy / 365)
    day_cos = np.cos(2 * np.pi * doy / 365)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return np.array([
        temp, humidity, precip, windspeed, winddir,
        cloudcover, dew, uvindex, sealevelpressure,
        doy, month, dow, day_sin, day_cos, month_sin, month_cos
    ]).reshape(1, -1)

def make_forecast(model, scaler, recent_block, base_date, horizon=7):
    X = recent_block.reshape(1, recent_block.shape[0], recent_block.shape[1])
    pred_scaled_flat = model.predict(X)
    pred_scaled = pred_scaled_flat.reshape(horizon, len(FEATURE_COLUMNS))

    dummy = np.zeros((horizon, scaler.n_features_in_))
    dummy[:, :len(FEATURE_COLUMNS)] = pred_scaled
    pred_unscaled_full = scaler.inverse_transform(dummy)
    pred_unscaled = pred_unscaled_full[:, :len(FEATURE_COLUMNS)]

    dates = [(base_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(horizon)]
    out = pd.DataFrame(pred_unscaled, columns=FEATURE_COLUMNS)
    out.insert(0, "date", dates)
    return out

def dewpoint_celsius(t_c, rh):
    a, b = 17.62, 243.12
    gamma = (a * t_c) / (b + t_c) + math.log(max(min(rh, 100.0), 0.1) / 100.0)
    return (b * gamma) / (a - gamma)

def icon_for_row(row):
    cc = row.get("cloudcover", 0)
    pr = row.get("precip", 0)
    ws = row.get("windspeed", 0)
    if pr >= 10: return "🌧️"
    if 0 < pr < 10: return "🌦️"
    if cc >= 80: return "☁️"
    if ws >= 30: return "💨"
    if cc <= 20: return "☀️"
    return "⛅"

def summary_for_row(date, row):
    icon = icon_for_row(row)
    temp = f"{row['temp']:.1f}°C"
    hum = f"{row['humidity']:.0f}% RH"
    wind = f"{row['windspeed']:.1f} km/h"
    precip = f"{row['precip']:.1f} mm"
    return f"{icon} **{date}** — {temp}, {hum}, wind {wind}, precip {precip}"

# ----------------------
# Farmer Advisor helpers (unchanged logic)
# ----------------------
def _weekly_stats_for_rules(df: pd.DataFrame) -> dict:
    wk = {}
    wk["rain_sum"]   = float(df["precip"].clip(lower=0).sum())
    wk["rain_mean"]  = float(df["precip"].clip(lower=0).mean())
    wk["tmax"]       = float(df["temp"].max())
    wk["tmin"]       = float(df["temp"].min())
    wk["tavg"]       = float(df["temp"].mean())
    wk["rh_mean"]    = float(df["humidity"].mean())
    wk["wind_max"]   = float(df["windspeed"].max())
    wk["cloud_mean"] = float(df["cloudcover"].mean())
    return wk
def farmer_decisions_colombo(forecast_df: pd.DataFrame, today=None) -> dict:
    """
    Lightweight heuristic guidance for Colombo (low-country wet zone).
    Not agronomic advice; for planning only.
    """
    today = today or datetime.now()
    wk = _weekly_stats_for_rules(forecast_df)

    notes, tasks, crops = [], [], []

    m = today.month
    in_maha = m in [9,10,11,12,1]   # Sep–Jan
    in_yala = m in [4,5,6,7,8]      # Apr–Aug

    # Rainfall
    if wk["rain_sum"] >= 70:
        notes.append("Weekly rainfall **high (≥70 mm)** — moisture adequate; watch waterlogging.")
    elif 30 <= wk["rain_sum"] < 70:
        notes.append("Weekly rainfall **moderate (30–70 mm)** — occasional irrigation may be needed.")
    else:
        notes.append("Weekly rainfall **low (<30 mm)** — plan **regular irrigation** or drought-tolerant crops.")

    # Temperature
    if wk["tavg"] >= 30:
        notes.append("Avg temp **≥30 °C** — heat stress risk; prefer short-duration/heat-tolerant varieties.")
    elif 26 <= wk["tavg"] < 30:
        notes.append("Avg temp **26–30 °C** — favorable for many vegetables and rice.")
    else:
        notes.append("Avg temp **<26 °C** — slower growth; choose tolerant/leafy crops.")

    # Wind
    if wk["wind_max"] >= 35:
        notes.append("Peak wind **≥35 km/h** — stake vines, add windbreaks, avoid tall/leggy transplants this week.")

    # Crop suggestions
    if wk["rain_sum"] >= 60 and (in_maha or m in [2,3]):
        crops += ["Paddy (if fields prepared)", "Taro/Arbi", "Okra (well-drained beds)"]
        tasks += ["Check drainage; avoid prolonged standing water on seedlings."]
    elif 30 <= wk["rain_sum"] < 60:
        crops += ["Leafy greens (kangkung, spinach)", "Long bean", "Okra", "Cucumber", "Banana/Plantain"]
        tasks += ["Mulch to conserve moisture; plan 1–2 irrigations in dry spells."]
    else:
        crops += ["Chili", "Eggplant/Brinjal", "Cowpea", "Lady’s finger (with irrigation)", "Cassava"]
        tasks += ["Install drip/sprinkler schedules; mulch heavily; sow early/late day."]

    # Cloud/UV hints
    if wk["cloud_mean"] >= 70:
        notes.append("Cloud cover **high** — monitor fungal disease; ensure row airflow.")
    if wk["cloud_mean"] <= 30 and wk["tavg"] >= 30:
        notes.append("Sunny + hot — consider **30–40% shade net** for nursery/leafy beds.")

    notes.append("Season hint: **Maha (Sep–Jan)** suits rice & many vegetables; **Yala (Apr–Aug)** favors short-cycle veg with irrigation.")

    return {
        "weekly": wk,
        "crops": sorted(set(crops)),
        "tasks": tasks,
        "notes": notes,
    }
# <<< Added for Farmer Advisor (original helpers)

# >>> Farmer Advisor — Modern helpers
# Colombo monthly climate normals (approximate; edit if you have official values)
CLIMATE_NORMALS = {
    1: {"rain": 60,  "tavg": 27.0},
    2: {"rain": 70,  "tavg": 27.2},
    3: {"rain": 130, "tavg": 28.0},
    4: {"rain": 250, "tavg": 28.6},
    5: {"rain": 370, "tavg": 28.3},
    6: {"rain": 230, "tavg": 27.8},
    7: {"rain": 120, "tavg": 27.5},
    8: {"rain": 120, "tavg": 27.4},
    9: {"rain": 240, "tavg": 27.3},
    10: {"rain": 360, "tavg": 27.4},
    11: {"rain": 310, "tavg": 27.2},
    12: {"rain": 170, "tavg": 27.0},
}

def compare_to_normals(weekly: dict, month: int) -> dict:
    """Return % difference vs monthly normals for rain and tavg."""
    norm = CLIMATE_NORMALS.get(month, {"rain": 150, "tavg": 27.5})
    rain_week_norm = norm["rain"] / 4.0
    rain_delta_pct = None if rain_week_norm == 0 else ((weekly["rain_sum"] - rain_week_norm) / max(rain_week_norm, 1e-6)) * 100.0
    tavg_delta = weekly["tavg"] - norm["tavg"]
    return {
        "rain_week_norm": rain_week_norm,
        "rain_delta_pct": rain_delta_pct,
        "tavg_norm": norm["tavg"],
        "tavg_delta": tavg_delta,
    }

def risk_scores(weekly: dict) -> dict:
    """
    Map weekly stats to 0–100 risk levels:
    - water: low rain -> high risk; extreme rain -> flood risk bump
    - heat: high average temp
    - wind: max wind
    - disease: combo of humidity, cloud, and rain
    """
    rain = weekly["rain_sum"]
    if rain < 30: water = 85
    elif rain < 60: water = 65
    elif rain <= 120: water = 35
    elif rain <= 180: water = 55
    else: water = 75  # flood/waterlogging risk

    t = weekly["tavg"]
    if t < 26: heat = 25
    elif t < 28: heat = 40
    elif t < 30: heat = 60
    else: heat = 80

    w = weekly["wind_max"]
    if w < 20: wind = 20
    elif w < 30: wind = 45
    elif w < 40: wind = 70
    else: wind = 85

    rh = weekly["rh_mean"]; cloud = weekly["cloud_mean"]
    disease_base = (0.5*(min(rh,100)/100) + 0.3*(cloud/100) + 0.2*min(rain/120, 1.5)) * 100
    disease = max(20, min(90, disease_base))
    return {"water": int(water), "heat": int(heat), "wind": int(wind), "disease": int(disease)}

def irrigation_plan(weekly: dict) -> dict:
    """Rough irrigation guidance (sessions/week & tip)."""
    rain = weekly["rain_sum"]
    if rain >= 80:
        return {"sessions": 0, "tip": "Skip routine irrigation; only spot-water raised beds."}
    if 40 <= rain < 80:
        return {"sessions": 1, "tip": "One light irrigation mid-week; mulch to retain moisture."}
    if 20 <= rain < 40:
        return {"sessions": 2, "tip": "Two irrigations (early morning); consider drip/sprinkler."}
    return {"sessions": 3, "tip": "Three irrigations; use drip + heavy mulch; prioritize young beds."}

def best_sowing_days(df: pd.DataFrame, n=2) -> list:
    """Pick n dates with low rain & moderate temps from the 7-day forecast."""
    tmp = df.copy()
    tmp["precip_norm"] = (tmp["precip"].max() - tmp["precip"]) / max(tmp["precip"].max(), 1e-6)
    tmp["t_score"] = 1 - (abs(tmp["temp"] - 28) / 10).clip(0, 1)
    tmp["score"] = 0.6*tmp["precip_norm"] + 0.4*tmp["t_score"]
    return tmp.nlargest(n, "score")["date"].tolist()

def crop_buckets(decisions_or_dict: dict) -> dict:
    """Split crops list into buckets for UI."""
    crops = decisions_or_dict.get("crops", [])
    cats = {"Rice": [], "Vegetables": [], "Fruits": [], "Spices/Roots": [], "Other": []}
    for c in crops:
        name = c.lower()
        if "paddy" in name or "rice" in name: cats["Rice"].append(c)
        elif any(k in name for k in ["okra","bean","cucumber","chili","eggplant","brinjal","cowpea","leafy","spinach","kangkung","lady"]):
            cats["Vegetables"].append(c)
        elif any(k in name for k in ["banana","plantain","fruit"]):
            cats["Fruits"].append(c)
        elif any(k in name for k in ["cassava","taro","arbi"]):
            cats["Spices/Roots"].append(c)
        else:
            cats["Other"].append(c)
    return cats

def badge_color(value, good_low=True):
    """Return a background color for numeric badges."""
    if good_low:
        if value <= 33: return "#16a34a22"
        if value <= 66: return "#f59e0b22"
        return "#dc262622"
    else:
        if value >= 66: return "#16a34a22"
        if value >= 33: return "#f59e0b22"
        return "#dc262622"
# <<< Farmer Advisor — Modern helpers

# >>> Farmer Advisor — Context-aware helpers (soil/irrigation/goal/IPM)
def pest_disease_watchlist(weekly: dict) -> list:
    """Signal likely problems from weather patterns (IPM-friendly; no chemical advice)."""
    items = []
    rain, rh, cloud, wind, tavg = weekly["rain_sum"], weekly["rh_mean"], weekly["cloud_mean"], weekly["wind_max"], weekly["tavg"]

    if rain >= 60 or (rh >= 85 and cloud >= 60):
        items += [
            "Fungal leaf spots & blights: ensure 30–40 cm row airflow, avoid overhead watering late day, remove infected leaves",
            "Snails/slugs after rain: use beer traps, hand-pick at dusk, keep beds tidy",
        ]
    if (tavg >= 28 and rain >= 50) or (rh >= 90):
        items += ["Bacterial soft rot/wilt risk in veg: avoid waterlogging, sanitize tools, rotate beds (avoid solanaceae repeats)"]
    if tavg >= 30 and cloud <= 30:
        items += ["Heat/sunscald on fruits/leafy beds: use 30–40% shade net mid-day, irrigate early morning"]
    if wind >= 35:
        items += ["Wind damage/lodging: stake vines/okra, install temporary windbreaks, avoid tall leggy transplants this week"]
    if rain < 30:
        items += ["Aphids/mites under dry heat: encourage beneficials, use yellow sticky traps, spot-spray soap solution on undersides"]

    seen = set(); out=[]
    for x in items:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def suggest_crops_with_context(base_list: list, soil: str, drainage: str, irrigation: str, goal: str) -> list:
    """Tailor generic crop list with field context."""
    s = soil.lower(); d = drainage.lower(); ir = irrigation.lower(); g = goal.lower()
    add, remove = set(), set()

    if s == "clay":
        add |= {"Paddy (if fields prepared)", "Taro/Arbi", "Banana/Plantain"}
        if d == "poor": remove |= {"Cucumber"}
    elif s == "loam":
        add |= {"Leafy greens (kangkung, spinach)", "Long bean", "Okra", "Eggplant/Brinjal"}
    elif s == "sandy":
        add |= {"Chili", "Cowpea", "Cassava", "Okra"}
        remove |= {"Paddy"}

    if d == "poor":
        add |= {"Raised-bed Leafy greens", "Okra (well-drained beds)"}
        remove |= {"Taro/Arbi"}
    elif d == "good":
        add |= {"Cucumber", "Long bean", "Eggplant/Brinjal"}

    if ir == "none":
        add |= {"Cassava", "Cowpea", "Chili"}
        remove |= {"Leafy greens (kangkung, spinach)", "Cucumber"}
    elif ir == "limited":
        add |= {"Okra", "Cowpea", "Chili", "Eggplant/Brinjal"}
    elif ir == "reliable":
        add |= {"Leafy greens (kangkung, spinach)", "Cucumber", "Long bean"}

    if g == "leafy/quick market":
        add |= {"Leafy greens (kangkung, spinach)", "Amaranth", "Coriander (herb)"}
    elif g == "fruiting veg market":
        add |= {"Chili", "Eggplant/Brinjal", "Okra", "Cucumber", "Tomato (if drainage good)"}
    elif g == "root/spice focus":
        add |= {"Cassava", "Taro/Arbi"}

    final = [c for c in base_list if c not in remove]
    for c in sorted(add):
        if c not in final:
            final.append(c)
    return final
# <<< Farmer Advisor — Context-aware helpers

# >>> Farmer Advisor — Localization + Paddy varieties
CROP_I18N = {
    "en": {
        "Paddy (if fields prepared)": "Paddy (if fields prepared)",
        "Taro/Arbi": "Taro/Arbi",
        "Okra (well-drained beds)": "Okra (well-drained beds)",
        "Okra": "Okra",
        "Leafy greens (kangkung, spinach)": "Leafy greens (kangkung, spinach)",
        "Long bean": "Long bean",
        "Cucumber": "Cucumber",
        "Banana/Plantain": "Banana/Plantain",
        "Chili": "Chili",
        "Eggplant/Brinjal": "Eggplant/Brinjal",
        "Cowpea": "Cowpea",
        "Lady’s finger (with irrigation)": "Lady’s finger (with irrigation)",
        "Cassava": "Cassava",
        "Raised-bed Leafy greens": "Raised-bed Leafy greens",
        "Paddy (transplanting if fields prepared)": "Paddy (transplanting if fields prepared)",
        "Paddy": "Paddy",
        "Amaranth": "Amaranth",
        "Coriander (herb)": "Coriander (herb)",
        "Tomato (if drainage good)": "Tomato (if drainage good)",
    },
    "si": {
        "Paddy (if fields prepared)": "වැවිලි පොළව සූදානම් නම් නෙල්",
        "Taro/Arbi": "කොළ අල (හබරලා/අර්බි)",
        "Okra (well-drained beds)": "බණ්ඩක්කා (දියනිරෝධිත ඇඳිලි)",
        "Okra": "බණ්ඩක්කා",
        "Leafy greens (kangkung, spinach)": "කිරි ඇළ, නිවිති ඇතුළු කොළ වර්ග",
        "Long bean": "මැස්සෝ",
        "Cucumber": "පිපිඤ්ඤා",
        "Banana/Plantain": "කෙසෙල්",
        "Chili": "මිරිස්",
        "Eggplant/Brinjal": "වම්බටු",
        "Cowpea": "කවුපි",
        "Lady’s finger (with irrigation)": "බණ්ඩක්කා (ණීරෙදා සමඟ)",
        "Cassava": "මැණිඔක්",
        "Raised-bed Leafy greens": "උස ඇඳිලි කොළ වර්ග",
        "Paddy (transplanting if fields prepared)": "ගමලෙන් වගා කිරීම සඳහා නෙල් (පොළව සූදානම් නම්)",
        "Paddy": "නෙල්",
        "Amaranth": "කුරුකරංදා/තම්බුට්ටි",
        "Coriander (herb)": "කොත්තමල්ලි (ඇට)",
        "Tomato (if drainage good)": "තක්කාලි (නෙරපාරුව හොඳනම්)",
    },
    "ta": {
        "Paddy (if fields prepared)": "நெல் (பயிரிடும் நிலம் தயார் என்றால்)",
        "Taro/Arbi": "சேப்பங்கிழங்கு/அர்பி",
        "Okra (well-drained beds)": "வெண்டைக்காய் (நன்றாக வடிகால் படுக்கை)",
        "Okra": "வெண்டைக்காய்",
        "Leafy greens (kangkung, spinach)": "கீரைகள் (கங்க்கொங், கீரை)",
        "Long bean": "அச்சி பயறு/காராமணி",
        "Cucumber": "வெள்ளரிக்காய்",
        "Banana/Plantain": "வாழை",
        "Chili": "மிளகாய்",
        "Eggplant/Brinjal": "கத்தரிக்காய்",
        "Cowpea": "கராமணி",
        "Lady’s finger (with irrigation)": "வெண்டை (நீர்ப்பாசனம் உடன்)",
        "Cassava": "மரவள்ளிக்கிழங்கு",
        "Raised-bed Leafy greens": "உயர்ந்த படுக்கை கீரைகள்",
        "Paddy (transplanting if fields prepared)": "நெல் (நடுக்குதல் — நிலம் தயார் என்றால்)",
        "Paddy": "நெல்",
        "Amaranth": "தண்டு கீரை/அமராந்த்",
        "Coriander (herb)": "கொத்தமல்லி (இலை/விதை)",
        "Tomato (if drainage good)": "தக்காளி (வடிகால் நன்றாக இருந்தால்)",
    },
}

def localize_crop_list(crops: list, lang: str) -> list:
    d = CROP_I18N.get(lang, CROP_I18N["en"])
    return [d.get(c, c) for c in crops]

PADDY_VARIETIES = [
    {"name": "BG 300", "duration": "Short (90–105d)", "days": 100, "season": "Both", "tol": ["drought"], "note": "Short-duration; suitable for Yala with limited water"},
    {"name": "BG 352", "duration": "Medium (105–120d)","days": 115, "season": "Both", "tol": ["flood"],   "note": "Tolerates brief waterlogging; sturdy stem"},
    {"name": "BG 360", "duration": "Medium (105–120d)","days": 115, "season": "Both", "tol": ["disease"], "note": "Good disease tolerance"},
    {"name": "BG 366", "duration": "Medium (105–120d)","days": 115, "season": "Maha", "tol": ["flood"],   "note": "Popular in wet season; good tillering"},
    {"name": "BG 380", "duration": "Medium (105–120d)","days": 118, "season": "Both", "tol": ["drought"], "note": "Performs under moderate water stress"},
    {"name": "BG 403", "duration": "Long (120–135d)",  "days": 125, "season": "Maha", "tol": ["flood"],   "note": "Long duration; stable yield"},
    {"name": "LD 371", "duration": "Short (90–105d)",  "days": 100, "season": "Yala", "tol": ["drought"], "note": "Short cycle; suits late planting"},
    {"name": "At 362", "duration": "Medium (105–120d)","days": 115, "season": "Both", "tol": ["salinity"],"note": "Better under mild salinity"},
]

def choose_paddy_varieties(weekly: dict, month: int, drainage: str, irrigation: str,
                           pref_duration: str, pref_tolerance: str, max_items=4) -> list:
    in_maha = month in [9,10,11,12,1]
    in_yala = month in [4,5,6,7,8]

    rain = weekly["rain_sum"]
    tavg = weekly["tavg"]
    need_tol = set()
    if rain >= 120 or drainage.lower() == "poor":
        need_tol.add("flood")
    if rain < 40 or irrigation.lower() in ["none", "limited"]:
        need_tol.add("drought")
    if tavg >= 29 and weekly["rh_mean"] >= 85:
        need_tol.add("disease")

    res = []
    for v in PADDY_VARIETIES:
        if in_maha and v["season"] not in ["Maha", "Both"]:
            continue
        if in_yala and v["season"] not in ["Yala", "Both"]:
            continue
        if pref_duration != "Any" and v["duration"].split()[0] not in pref_duration:
            continue
        if pref_tolerance != "Any" and pref_tolerance not in v["tol"]:
            continue
        if need_tol and not (need_tol & set(v["tol"])):
            continue
        res.append(v)

    if not res:
        for v in PADDY_VARIETIES:
            if pref_duration != "Any" and v["duration"].split()[0] not in pref_duration:
                continue
            if pref_tolerance != "Any" and pref_tolerance not in v["tol"]:
                continue
            res.append(v)
    if not res:
        res = PADDY_VARIETIES[:]
    return res[:max_items]

UI_I18N = {
    "en": {"recommended_crops": "Recommended Crops (by category, tailored)", "tasks": "Field Tasks / Practices (auto + context)",
           "notes": "Notes & Risks", "ipm": "IPM Watchlist (this week)", "paddy_title": "Paddy varieties (based on season/climate & your preferences)",
           "duration": "Duration", "tolerance": "Tolerance", "season": "Season", "days": "Days", "note": "Note"},
    "si": {"recommended_crops": "නිර්දේශිත බෝග (ප්‍රවර්ග අනුව, ක්ෂේත්‍රයට ගැළපේ)", "tasks": "ක්ෂේත්‍ර කාර්යයන් / ක්‍රියාමාර්ග (ස්වයං + පසුබිම)",
           "notes": "සටහන් & අවදානම්", "ipm": "IPM අවධානම් ලැයිස්තුව (මෙම සතිය)", "paddy_title": "නෙල් වර්ග (රිතූ/කාලගුණය සහ ඔබේ කැමැත්ත අනුව)",
           "duration": "කාලය", "tolerance": "ඉවසීම", "season": "රිතූ", "days": "දිනයන්", "note": "සටහන"},
    "ta": {"recommended_crops": "பரிந்துரைக்கப்பட்ட பயிர்கள் (வகைப்படி, தளத்திற்கு ஏற்றது)", "tasks": "புல வேலைகள் / நடைமுறைகள் (தானாக + சூழல்)",
           "notes": "குறிப்புகள் & அபாயங்கள்", "ipm": "IPM கண்காணிப்பு (இந்த வாரம்)", "paddy_title": "நெல் வகைகள் (ருது/காலநிலை & உங்கள் முன்னுரிமை அடிப்படையில்)",
           "duration": "நாட்கள்-நீளம்", "tolerance": "தாங்குதல்", "season": "ருது", "days": "நாட்கள்", "note": "குறிப்பு"},
}
# <<< Farmer Advisor — Localization + Paddy varieties

# ----------------------
# Sidebar: model + live weather controls
# ----------------------
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    try:
        model, scaler, model_path_used, scaler_path_used = load_model_and_scaler()
        st.success("Model loaded", icon="✅")
        st.caption(f"**Model:** `{os.path.basename(model_path_used)}`")
        st.caption(f"**Scaler:** `{os.path.basename(scaler_path_used)}`")
    except Exception as e:
        st.error(f"Failed to load model/scaler:\n{e}")
        st.stop()

    st.markdown("### 🌍 Live Weather (optional)")
    api_key = st.text_input("OpenWeatherMap API key", type="password", help="Get a free key at openweathermap.org")
    city = st.text_input("City (e.g., Colombo, London)")
    use_live = st.button("📡 Fetch current weather")

    st.markdown("### 🔄 Input Mode")
    mode = st.radio(
        "Choose how you’ll provide data",
        ["Enter today's values", "Upload last 30 days (CSV)"],
        help="Uploading 30 days gives better context to the LSTM."
    )

    show_advanced = st.toggle("Show advanced chart settings", value=False)

# Prepare session state for inputs so we can auto-fill after API fetch
default_values = {
    "temp": 25.0, "humidity": 75.0, "precip": 5.0,
    "windspeed": 10.0, "winddir": 180.0, "cloudcover": 60.0,
    "dew": 18.0, "uvindex": 6.0, "sealevelpressure": 1012.0
}
for k, v in default_values.items():
    st.session_state.setdefault(k, v)

# ----------------------
# Live fetch handler
# ----------------------
city_latlon = None
if use_live:
    if not api_key or not city:
        st.sidebar.error("Please enter both API key and City.")
    else:
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {"q": city, "appid": api_key, "units": "metric"}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()

            main = data.get("main", {})
            wind = data.get("wind", {})
            clouds = data.get("clouds", {})
            rain = data.get("rain", {})
            snow = data.get("snow", {})

            temp = float(main.get("temp", st.session_state["temp"]))
            humidity = float(main.get("humidity", st.session_state["humidity"]))
            pressure = float(main.get("pressure", st.session_state["sealevelpressure"]))
            wind_speed_ms = float(wind.get("speed", 0.0))
            wind_speed_kmh = wind_speed_ms * 3.6
            wind_dir = float(wind.get("deg", st.session_state["winddir"]))
            cloud_pct = float(clouds.get("all", st.session_state["cloudcover"]))

            precip = float(rain.get("1h", 0.0) or snow.get("1h", 0.0) or 0.0)
            dew = dewpoint_celsius(temp, humidity)

            uv = st.session_state["uvindex"]
            coord = data.get("coord", {})
            lat, lon = coord.get("lat"), coord.get("lon")
            city_latlon = (lat, lon)
            if lat is not None and lon is not None:
                try:
                    one_url = "https://api.openweathermap.org/data/3.0/onecall"
                    one_params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric", "exclude": "minutely,hourly,daily,alerts"}
                    r2 = requests.get(one_url, params=one_params, timeout=10)
                    if r2.ok:
                        uv = float(r2.json().get("current", {}).get("uvi", uv))
                except Exception:
                    pass

            st.session_state.update({
                "temp": round(temp, 1),
                "humidity": round(humidity, 0),
                "precip": round(precip, 1),
                "windspeed": round(wind_speed_kmh, 1),
                "winddir": round(wind_dir, 0),
                "cloudcover": round(cloud_pct, 0),
                "dew": round(dew, 1),
                "uvindex": round(uv, 1),
                "sealevelpressure": round(pressure, 0),
                "city_latlon": city_latlon,
                "city_name": data.get("name", city),
            })
            st.sidebar.success("Live weather fetched. Inputs updated ✅")
            st.rerun()
        except requests.HTTPError as he:
            st.sidebar.error(f"API error: {he.response.status_code} — {he.response.text[:120]}")
        except Exception as ex:
            st.sidebar.error(f"Failed to fetch weather: {ex}")

st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ----------------------
# Layout: inputs | output + map
# ----------------------
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown("#### 📥 Provide Input")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploader_placeholder = None
    history_df = None

    if mode == "Upload last 30 days (CSV)":
        st.write("CSV must include: **`datetime`** + " + ", ".join(f"`{c}`" for c in FEATURE_COLUMNS))
        uploader_placeholder = st.file_uploader("Upload CSV (≥ 30 recent rows recommended)", type=["csv"])
        if uploader_placeholder is not None:
            try:
                raw = uploader_placeholder.read()
                history_df = pd.read_csv(io.BytesIO(raw))
                if "datetime" not in history_df.columns:
                    st.error("CSV must contain a `datetime` column.")
                    history_df = None
                else:
                    history_df = add_date_features(history_df)
                    miss = [c for c in FEATURE_COLUMNS if c not in history_df.columns]
                    if miss:
                        st.error(f"Missing required columns: {miss}")
                        history_df = None
                    else:
                        st.success("CSV parsed successfully.", icon="📄")
            except Exception as ex:
                st.error(f"Failed to parse CSV: {ex}")
                history_df = None

    if mode == "Enter today's values":
        c1, c2, c3 = st.columns(3)
        temp = c1.number_input("Temperature (°C)", value=float(st.session_state["temp"]), key="temp")
        humidity = c2.number_input("Humidity (%)", value=float(st.session_state["humidity"]), min_value=0.0, max_value=100.0, key="humidity")
        precip = c3.number_input("Precipitation (mm)", value=float(st.session_state["precip"]), min_value=0.0, key="precip")

        c4, c5, c6 = st.columns(3)
        windspeed = c4.number_input("Wind Speed (km/h)", value=float(st.session_state["windspeed"]), min_value=0.0, key="windspeed")
        winddir = c5.number_input("Wind Direction (°)", value=float(st.session_state["winddir"]), min_value=0.0, max_value=360.0, key="winddir")
        cloudcover = c6.number_input("Cloud Cover (%)", value=float(st.session_state["cloudcover"]), min_value=0.0, max_value=100.0, key="cloudcover")

        c7, c8, c9 = st.columns(3)
        dew = c7.number_input("Dew Point (°C)", value=float(st.session_state["dew"]), key="dew")
        uvindex = c8.number_input("UV Index", value=float(st.session_state["uvindex"]), min_value=0.0, key="uvindex")
        sealevelpressure = c9.number_input("Sea Level Pressure (hPa)", value=float(st.session_state["sealevelpressure"]), key="sealevelpressure")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown("#### 🔮 Forecast")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    go = st.button("🚀 Predict Next 7 Days", type="primary", use_container_width=True)

    if go:
        try:
            if mode == "Upload last 30 days (CSV)" and history_df is not None:
                if len(history_df) < 30:
                    st.warning("Need at least 30 recent rows. Using last available rows with repetition.")
                block = history_df.tail(30).copy()
                while len(block) < 30:
                    block = pd.concat([history_df.tail(1), block], ignore_index=True)

                base_date = pd.to_datetime(history_df["datetime"].iloc[-1])
                recent = block[ALL_FEATURES].values
                recent_scaled = scaler.transform(recent)
                forecast_df = make_forecast(model, scaler, recent_scaled, base_date, horizon=7)
            else:
                today_features = build_today_row(
                    st.session_state["temp"], st.session_state["humidity"], st.session_state["precip"],
                    st.session_state["windspeed"], st.session_state["winddir"],
                    st.session_state["cloudcover"], st.session_state["dew"],
                    st.session_state["uvindex"], st.session_state["sealevelpressure"],
                    now=datetime.now()
                )
                recent_scaled = scaler.transform(today_features)
                recent_scaled_30 = np.repeat(recent_scaled, 30, axis=0)
                base_date = datetime.now()
                forecast_df = make_forecast(model, scaler, recent_scaled_30, base_date, horizon=7)

            # Save forecast for Advisor section
            st.session_state["forecast_df"] = forecast_df.copy()

            # KPI Row
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"""<div class="kpi"><span class="label">Day 1 Temp (°C)</span>
            <span class="value">{forecast_df.loc[0,'temp']:.1f}</span></div>""", unsafe_allow_html=True)
            k2.markdown(f"""<div class="kpi"><span class="label">Day 1 Humidity (%)</span>
            <span class="value">{forecast_df.loc[0,'humidity']:.0f}</span></div>""", unsafe_allow_html=True)
            k3.markdown(f"""<div class="kpi"><span class="label">Day 1 Precip (mm)</span>
            <span class="value">{forecast_df.loc[0,'precip']:.1f}</span></div>""", unsafe_allow_html=True)
            k4.markdown(f"""<div class="kpi"><span class="label">Day 1 Wind (km/h)</span>
            <span class="value">{forecast_df.loc[0,'windspeed']:.1f}</span></div>""", unsafe_allow_html=True)

            # Tabs
            t1, t2, t3, t4 = st.tabs(["📈 Charts", "📋 Table", "📝 Daily summaries", "⬇️ Export"])

            with t1:
                template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
                plot_df = forecast_df.melt(id_vars="date", value_vars=FEATURE_COLUMNS, var_name="feature", value_name="value")
                if show_advanced:
                    st.caption("Tip: Click legend to isolate a series. Drag to zoom; double-click to reset.")
                fig = px.line(plot_df, x="date", y="value", color="feature", markers=True,
                              title="7-Day Forecast • All Features", template=template)
                fig.update_layout(hovermode="x unified", legend_title_text="Feature")
                st.plotly_chart(fig, use_container_width=True, theme=None)

            with t2:
                st.dataframe(forecast_df.round(2), use_container_width=True, hide_index=True)

            with t3:
                for _, r in forecast_df.iterrows():
                    st.markdown(f"""<div class="daycard">
                        <div>{summary_for_row(r['date'], r)}</div>
                        <div class="daymeta">cloud {r['cloudcover']:.0f}% • UV {r['uvindex'] if 'uvindex' in r else '-'} • pressure {r['sealevelpressure'] if 'sealevelpressure' in r else '-'} hPa</div>
                    </div>""", unsafe_allow_html=True)

            with t4:
                csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", data=csv_bytes, file_name="forecast_7day.csv",
                                   mime="text/csv", use_container_width=True)
                st.caption("The file includes date + all predicted features.")

            st.success("Prediction complete. Adjust inputs or upload a different file to re-run.", icon="🎉")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("Tip: Ensure your model & scaler were trained with the same feature order as this app expects.")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------
# Map (shows if we have coordinates from live fetch)
# ----------------------
loc = st.session_state.get("city_latlon")
name = st.session_state.get("city_name")
if loc and all(loc):
    st.markdown("#### 🗺️ Location")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    lat, lon = loc
    df_map = pd.DataFrame({"lat": [lat], "lon": [lon], "city": [name or ""]})
    st.map(df_map, latitude="lat", longitude="lon", size=50, color=None, zoom=9)
    st.caption(f"City: **{name}**  •  Lat: {lat:.4f}, Lon: {lon:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------
# 📌 Decision Center — Colombo (ENHANCED Farmer + Business Advisors)
# ----------------------
st.markdown("#### 📌 Decision Center — Colombo")
st.markdown('<div class="card">', unsafe_allow_html=True)

forecast_ready = "forecast_df" in st.session_state and isinstance(st.session_state["forecast_df"], pd.DataFrame)
city_name = (st.session_state.get("city_name") or "").strip()

if not forecast_ready:
    st.warning("Run the forecast first (click **Predict Next 7 Days**).", icon="⏱️")
else:
    import json

    df7 = st.session_state["forecast_df"].copy()
    wk = _weekly_stats_for_rules(df7)
    month_now = datetime.now().month
    cmp = compare_to_normals(wk, month_now)
    risks = risk_scores(wk)
    sow_days = best_sowing_days(df7, n=2)

    if city_name and city_name.lower() != "colombo":
        st.info(f"Using **{city_name}** weather; rules are tuned for **Colombo** (low-country wet zone).", icon="ℹ️")

    # ====== Top Snapshot ======
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.markdown(f'<div class="badge">Rain (7d): <b>{wk["rain_sum"]:.0f} mm</b></div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="badge">Temp avg: <b>{wk["tavg"]:.1f} °C</b></div>', unsafe_allow_html=True)
    s3.markdown(f'<div class="badge">RH avg: <b>{wk["rh_mean"]:.0f}%</b></div>', unsafe_allow_html=True)
    s4.markdown(f'<div class="badge">Wind max: <b>{wk["wind_max"]:.0f} km/h</b></div>', unsafe_allow_html=True)
    rain_delta = "—"
    if cmp["rain_delta_pct"] is not None:
        sign = "+" if cmp["rain_delta_pct"] >= 0 else ""
        rain_delta = f'{sign}{cmp["rain_delta_pct"]:.0f}% vs wk-normal'
    s5.markdown(f'<div class="badge">Climate: rain {rain_delta}</div>', unsafe_allow_html=True)
    s6.markdown(f'<div class="badge">ΔT vs normal: <b>{cmp["tavg_delta"]:+.1f}°C</b></div>', unsafe_allow_html=True)

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)

    # ====== Shared "What-if" Scenario controls ======
    with st.expander("🔧 What-if Scenario (applies to both advisors)"):
        adj1, adj2, adj3 = st.columns([1,1,1])
        rain_pct = adj1.slider("Adjust rainfall (%)", -50, 50, 0, help="Scale daily precip by this percentage")
        temp_delta = adj2.slider("Adjust temperature (°C)", -5, 5, 0, help="Shift daily temp by this °C")
        wind_pct = adj3.slider("Adjust wind (%)", -50, 50, 0, help="Scale daily windspeed by this percentage")

        df7_scn = df7.copy()
        df7_scn["precip"] = (df7_scn["precip"] * (1 + rain_pct/100.0)).clip(lower=0)
        df7_scn["temp"] = df7_scn["temp"] + temp_delta
        df7_scn["windspeed"] = (df7_scn["windspeed"] * (1 + wind_pct/100.0)).clip(lower=0)
        wk_scn = _weekly_stats_for_rules(df7_scn)
        risks_scn = risk_scores(wk_scn)
        sow_days_scn = best_sowing_days(df7_scn, n=2)

        st.caption(
            f"Scenario → Rain: **{wk_scn['rain_sum']:.0f} mm** • Tavg: **{wk_scn['tavg']:.1f}°C** • "
            f"Wind max: **{wk_scn['wind_max']:.0f} km/h** | Risks → Water **{risks_scn['water']}**, "
            f"Heat **{risks_scn['heat']}**, Wind **{risks_scn['wind']}**, Disease **{risks_scn['disease']}**"
        )

    # ====== Tabs: Farmer | Business | Print/Export ======
    t_farmer, t_business, t_export = st.tabs(["🌾 Farmer Advisor", "🏢 Business Advisor", "🖨️ Project Panel / Export"])

    # =========================
    # 🌾 FARMER ADVISOR (Enhanced)
    # =========================
    with t_farmer:
        # Context selectors
        cc1, cc2, cc3, cc4 = st.columns(4)
        soil = cc1.selectbox("Soil", ["Clay", "Loam", "Sandy"], index=1)
        drainage = cc2.selectbox("Drainage", ["Poor", "Moderate", "Good"], index=1)
        irrigation = cc3.selectbox("Irrigation", ["None", "Limited", "Reliable"], index=1)
        goal = cc4.selectbox("Goal", ["Subsistence mix", "Leafy/quick market", "Fruiting veg market", "Root/spice focus"], index=0)

        cc5, cc6, cc7 = st.columns([1,1,1])
        lang = cc5.selectbox("Language", ["en", "si", "ta"], index=0, help="en=English, si=සිංහල, ta=தமிழ்")
        pref_duration = cc6.selectbox("Paddy duration", ["Any", "Short", "Medium", "Long"], index=0)
        pref_tolerance = cc7.selectbox("Paddy tolerance", ["Any", "flood", "drought", "disease", "salinity"], index=0)

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)

        # Risk bars (scenario-aware toggle)
        use_scenario = st.toggle("Use scenario-adjusted risks & plan", value=False)
        wk_use = wk_scn if use_scenario else wk
        risks_use = risks_scn if use_scenario else risks
        df_use = df7_scn if use_scenario else df7
        sow_use = sow_days_scn if use_scenario else sow_days

        st.write("**Risk overview (0 = low, 100 = high)**")
        for label, val in [("Water", risks_use["water"]), ("Heat", risks_use["heat"]),
                           ("Wind", risks_use["wind"]), ("Disease", risks_use["disease"])]:
            st.markdown(
                f'''
                <div style="margin:6px 0;">
                  <div style="display:flex;justify-content:space-between;">
                    <span>{label}</span><span>{val}</span>
                  </div>
                  <div style="height:10px;border-radius:999px;background:#e5e7eb;">
                    <div style="width:{val}%;height:10px;border-radius:999px;background:{badge_color(val, good_low=True)};"></div>
                  </div>
                </div>
                ''',
                unsafe_allow_html=True
            )
        with st.expander("Why these risk scores?"):
            st.write(
                f"- **Water**: Rain {wk_use['rain_sum']:.0f} mm/7d (60–120 comfy; low → drought; very high → waterlogging)\n"
                f"- **Heat**: Avg {wk_use['tavg']:.1f} °C (≥30 °C = stress)\n"
                f"- **Wind**: Peak {wk_use['wind_max']:.0f} km/h (≥35 km/h stake vines / windbreaks)\n"
                f"- **Disease**: RH {wk_use['rh_mean']:.0f}%, cloud {wk_use['cloud_mean']:.0f}%, rain {wk_use['rain_sum']:.0f} mm"
            )

        # Core decisions
        decisions = farmer_decisions_colombo(df_use)
        tailored_crops = suggest_crops_with_context(decisions["crops"], soil, drainage, irrigation, goal)
        localized_crops = localize_crop_list(tailored_crops, lang)
        buckets = crop_buckets({"crops": localized_crops})

        a1, a2 = st.columns(2)
        with a1:
            title_i18n = UI_I18N[lang]["recommended_crops"]
            st.markdown(f"**{title_i18n}**")
            any_items = False
            for cat, items in buckets.items():
                if items:
                    any_items = True
                    st.markdown(f"- **{cat}:** " + ", ".join(items))
            if not any_items:
                st.write("—")

            st.markdown("—")
            st.markdown(f"**{UI_I18N[lang]['paddy_title']}**")
            pv = choose_paddy_varieties(
                weekly=wk_use, month=month_now, drainage=drainage, irrigation=irrigation,
                pref_duration=pref_duration, pref_tolerance=pref_tolerance, max_items=4
            )
            if pv:
                for v in pv:
                    st.markdown(
                        f"- **{v['name']}** — {UI_I18N[lang]['duration']}: *{v['duration']}*, "
                        f"{UI_I18N[lang]['season']}: *{v['season']}*, "
                        f"{UI_I18N[lang]['tolerance']}: *{', '.join(v['tol'])}*, "
                        f"{UI_I18N[lang]['days']}: *{v['days']}*. "
                        f"{UI_I18N[lang]['note']}: {v['note']}"
                    )
            else:
                st.write("—")

        with a2:
            st.markdown(f"**{UI_I18N[lang]['tasks']}**")
            extra_tasks = []
            if drainage.lower() == "poor":
                extra_tasks += ["Use ridges/raised beds; add sand/compost", "Keep field drains clear after heavy rain"]
            if irrigation.lower() in ["none", "limited"]:
                extra_tasks += ["Mulch 5–8 cm; irrigate early morning", "Prioritize seedlings & flowering plants for water"]
            if soil.lower() == "sandy":
                extra_tasks += ["Add compost/manure to improve water retention", "Windbreaks help reduce evapotranspiration"]
            tasks_final = decisions["tasks"] + extra_tasks
            st.write("• " + "\n• ".join(dict.fromkeys(tasks_final)) if tasks_final else "—")

            st.markdown(f"**{UI_I18N[lang]['notes']}**")
            st.write("• " + "\n• ".join(decisions["notes"]) if decisions["notes"] else "—")

            st.markdown(f"**{UI_I18N[lang]['ipm']}**")
            ipm = pest_disease_watchlist(wk_use)
            st.write("• " + "\n• ".join(ipm) if ipm else "—")

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)

        # Action Plan & Water Estimator
        b1, b2 = st.columns([1,1])
        with b1:
            st.markdown("**🗓️ 7-Day Field Action Plan**")
            # Per-day line from df_use
            for _, r in df_use.iterrows():
                flags = []
                if r["precip"] >= 10: flags.append("heavy rain")
                elif r["precip"] > 0: flags.append("showers")
                if r["temp"] >= 30: flags.append("hot")
                if r["windspeed"] >= 35: flags.append("windy")
                if r["cloudcover"] <= 30 and r["precip"] < 10: flags.append("clear")
                label = ", ".join(flags) if flags else "normal"
                st.markdown(
                    f"""<div class="daycard">
                        <div>📅 <b>{r['date']}</b> — {label.title()}</div>
                        <div class="daymeta">T {r['temp']:.1f}°C • P {r['precip']:.1f} mm • W {r['windspeed']:.0f} km/h</div>
                    </div>""",
                    unsafe_allow_html=True
                )
            st.caption("Tip: Do land prep/transplant on clearer, cooler days; avoid heavy-rain days for pesticide/fertilizer applications.")

        with b2:
            st.markdown("**💧 Irrigation Planner**")
            plan = irrigation_plan(wk_use)
            area = st.number_input("Field area (m²)", min_value=10.0, value=100.0, step=10.0)
            mm_per_session = st.slider("Water depth per session (mm)", 5, 30, 12)
            sessions = plan["sessions"]
            liters_per_session = area * mm_per_session  # 1 mm over 1 m² = 1 liter
            total_liters = sessions * liters_per_session
            st.write(
                f"- **Recommended sessions:** `{sessions} / week` — {plan['tip']}\n"
                f"- **Water per session:** ~**{liters_per_session:,.0f} L**\n"
                f"- **Total this week:** ~**{total_liters:,.0f} L**"
            )
            cost_per_1000L = st.number_input("Water cost per 1000 L (LKR)", min_value=0.0, value=80.0, step=5.0)
            est_cost = (total_liters/1000.0) * cost_per_1000L
            st.write(f"- **Estimated water cost:** ~**LKR {est_cost:,.0f}**")

        # Downloads for Farmer
        export_payload_farmer = {
            "weekly_stats": wk_use, "risks": risks_use, "scenario_used": use_scenario,
            "context": {"soil": soil, "drainage": drainage, "irrigation": irrigation, "goal": goal, "language": lang},
            "crops_by_category": buckets, "paddy_varieties": pv,
            "tasks": list(dict.fromkeys(tasks_final)), "notes": decisions["notes"], "ipm_watchlist": ipm,
            "sowing_days": sow_use, "irrigation": {"sessions": sessions, "mm_per_session": mm_per_session,
                                                   "area_m2": area, "water_liters_total": total_liters}
        }
        st.download_button(
            "⬇️ Download Farmer Plan (JSON)",
            data=json.dumps(export_payload_farmer, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="colombo_farmer_plan.json",
            mime="application/json",
            use_container_width=True
        )

        # Calendar (.ics) for best sowing days
        def _ics_for_days(summary_prefix, days):
            def as_ics_date(d): return d.replace("-", "")
            events = []
            for d in days:
                events.append(
f"""BEGIN:VEVENT
DTSTART;VALUE=DATE:{as_ics_date(d)}
DTEND;VALUE=DATE:{as_ics_date(d)}
SUMMARY:{summary_prefix} {d}
END:VEVENT"""
                )
            return "BEGIN:VCALENDAR\nVERSION:2.0\n" + "\n".join(events) + "\nEND:VCALENDAR\n"

        ics_farmer = _ics_for_days("Farm: Best sowing/transplant day", sow_use)
        st.download_button("🗓️ Add Sowing Days to Calendar (.ics)", data=ics_farmer.encode("utf-8"),
                           file_name="farmer_sowing_days.ics", mime="text/calendar", use_container_width=True)

    # =========================
    # 🏢 BUSINESS ADVISOR (Enhanced)
    # =========================
    with t_business:
        # Day-level flags & scores (scenario-aware)
        df_calc = (df7_scn if st.session_state.get("use_biz_scn", False) or use_scenario else df7).copy()
        df_calc["is_rainy"] = df_calc["precip"] >= 10
        df_calc["is_showery"] = (df_calc["precip"] > 0) & (df_calc["precip"] < 10)
        df_calc["is_hot"] = df_calc["temp"] >= 30
        df_calc["is_windy"] = df_calc["windspeed"] >= 35
        df_calc["is_clear"] = (df_calc["cloudcover"] <= 30) & (~df_calc["is_rainy"])

        rainy_days = int(df_calc["is_rainy"].sum())
        hot_days = int(df_calc["is_hot"].sum())
        windy_days = int(df_calc["is_windy"].sum())
        clear_days = int(df_calc["is_clear"].sum())
        hottest = float(df_calc["temp"].max())
        wettest = float(df_calc["precip"].max())

        tmp = df_calc.copy()
        tmp["rain_score"] = (tmp["precip"].max() - tmp["precip"]) / max(tmp["precip"].max(), 1e-6)
        tmp["temp_score"] = 1 - (abs(tmp["temp"] - 28) / 10).clip(0, 1)
        tmp["footfall_score"] = 0.65*tmp["rain_score"] + 0.35*tmp["temp_score"]
        best_days = tmp.nlargest(2, "footfall_score")["date"].tolist()

        # Badges
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.markdown(f'<div class="badge">Rain (7d): <b>{wk["rain_sum"]:.0f} mm</b></div>', unsafe_allow_html=True)
        b2.markdown(f'<div class="badge">Rainy days (≥10mm): <b>{rainy_days}</b></div>', unsafe_allow_html=True)
        b3.markdown(f'<div class="badge">Hot days (≥30°C): <b>{hot_days}</b></div>', unsafe_allow_html=True)
        b4.markdown(f'<div class="badge">Windy days (≥35 km/h): <b>{windy_days}</b></div>', unsafe_allow_html=True)
        b5.markdown(f'<div class="badge">Peak T / rain: <b>{hottest:.1f}°C</b> / <b>{wettest:.1f} mm</b></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        biz_type = c1.selectbox(
            "Business type",
            ["Retail shop", "Restaurant/Café", "Construction/Building", "Outdoor Events", "Logistics/Delivery"],
            index=0
        )
        venue = c2.selectbox("Venue", ["Indoor", "Semi-open", "Open"], index=0)
        priority = c3.selectbox("This week’s priority", ["Revenue boost", "Cost control", "Reliability/Continuity"], index=0)

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)

        # Risk bars (reuse farmer risk visuals)
        st.write("**Risk overview (0 = low, 100 = high)**")
        r_use = risks_scn if use_scenario else risks
        for label, val in [("Weather disruption", r_use["water"]),
                           ("Heat/Comfort strain", r_use["heat"]),
                           ("Wind/Outdoor safety", r_use["wind"]),
                           ("Hygiene/Disease (food handling)", r_use["disease"])]:
            st.markdown(
                f'''
                <div class="progress-wrap">
                  <div style="display:flex;justify-content:space-between;">
                    <span>{label}</span><span>{val}</span>
                  </div>
                  <div class="progress">
                    <div class="bar" style="width:{val}%;background:{badge_color(val, good_low=True)};"></div>
                  </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

        # Outlook
        footfall_msg = "mixed"
        if rainy_days >= 3:
            footfall_msg = "depressed during heavy-rain days; pivot to delivery/online"
        elif clear_days >= 3 and hot_days <= 2:
            footfall_msg = "favorable; plan in-store promos"
        elif hot_days >= 3:
            footfall_msg = "heat-sensitive; promote cold/indoor comfort"
        st.write(
            f"- **Footfall outlook:** {footfall_msg}\n"
            f"- **Supply/transport risk:** {'elevated' if rainy_days >= 2 or windy_days >= 1 else 'normal'}\n"
            f"- **Staff comfort:** {'watch heat stress' if hot_days >= 2 else 'normal'}\n"
            f"- **Energy demand:** {'likely higher (cooling)' if wk['tavg'] >= 29 else 'normal'}"
        )
        if city_name and city_name.lower() != "colombo":
            st.info(f"Using **{city_name}** weather; rules tuned for **Colombo**.", icon="ℹ️")

        # Recommendations (dynamic)
        def biz_tips():
            tips = []
            if biz_type == "Retail shop":
                if rainy_days >= 2:
                    tips += ["Push **online/WhatsApp orders** + same-day delivery",
                             "Place **umbrella/mat stands**; keep entrance dry (slip risk)",
                             "Bundle **rain gear** near checkout"]
                if clear_days >= 2:
                    tips += ["Run **in-store events** on best days: " + (", ".join(best_days) if best_days else "—")]
                if hot_days >= 2:
                    tips += ["Entrance **cooling products**; offer **cold water** station"]
            elif biz_type == "Restaurant/Café":
                if rainy_days >= 2:
                    tips += ["Promote **comfort menus** on apps; **free delivery above LKR X**",
                             "Check **leaks/canopies**; minimize outdoor seating"]
                if hot_days >= 2:
                    tips += ["Promote **cold beverages/desserts**; **happy hour** 2–5 pm"]
                if venue != "Indoor" and windy_days >= 1:
                    tips += ["Secure **awnings/signage**; move light furniture indoors"]
            elif biz_type == "Construction/Building":
                if rainy_days >= 2:
                    tips += ["Prioritize **indoor tasks**; schedule concrete on **drier windows**",
                             "Cover materials; improve **site drainage**; secure pumps"]
                if windy_days >= 1:
                    tips += ["Limit **heights/cranage**; reinforce temporary structures"]
                if hot_days >= 2:
                    tips += ["**Hydration & rest cycles**; shift heavy tasks to morning"]
            elif biz_type == "Outdoor Events":
                if rainy_days >= 1 or windy_days >= 1:
                    tips += ["**Tented cover** + **non-slip flooring**; have **Plan-B**",
                             "Moisture-proof **audio/electrical**; confirm vendor arrivals early"]
                if clear_days >= 2:
                    tips += ["Concentrate marketing to **best days**: " + (", ".join(best_days) if best_days else "—")]
            elif biz_type == "Logistics/Delivery":
                if rainy_days >= 2:
                    tips += ["Increase **rider slots**; extend **delivery windows** 15–20%",
                             "Weather-proof parcels; **route around flooded hotspots**"]
                if hot_days >= 2:
                    tips += ["Add **cooling breaks**; insulate **perishables**"]
            if priority == "Cost control":
                tips += ["Trim **AC setpoints** by 1°C + ceiling fans", "Align staffing to **best-demand days**"]
            elif priority == "Revenue boost":
                tips += ["Create **weather-timed offers** (rain → ‘Hot Drinks −15%’)",
                         "Send **SMS/WhatsApp** promo the evening before **best days**"]
            else:
                tips += ["Pre-position **inventory** before heavy-rain days", "Confirm **backup power** readiness"]
            # de-duplicate
            out, seen = [], set()
            for t in tips:
                if t not in seen: out.append(t); seen.add(t)
            return out

        recs = biz_tips()
        st.markdown("**✅ Recommendations**")
        st.write("• " + "\n• ".join(recs) if recs else "—")

        st.markdown("**🗓️ Operate smart by day**")
        for _, r in df_calc.iterrows():
            flags = []
            if r["is_rainy"]: flags.append("heavy rain")
            elif r["is_showery"]: flags.append("showers")
            if r["is_hot"]: flags.append("hot")
            if r["is_windy"]: flags.append("windy")
            if r["is_clear"]: flags.append("clear")
            lbl = ", ".join(flags) if flags else "normal"
            st.markdown(
                f"""<div class="daycard">
                    <div>📅 <b>{r['date']}</b> — {lbl.title()}</div>
                    <div class="daymeta">T {r['temp']:.1f}°C • P {r['precip']:.1f} mm • W {r['windspeed']:.0f} km/h</div>
                </div>""",
                unsafe_allow_html=True
            )
        st.markdown(
            f"- **Best promo/footfall days:** {', '.join(best_days) if best_days else '—'}  \n"
            f"- **Caution days (≥10 mm rain):** {', '.join(df_calc.loc[df_calc['is_rainy'],'date'].tolist()) or '—'}"
        )

        # Quick ROI / Revenue model
        st.markdown("**📈 Quick Revenue Model (7-day)**")
        m1, m2, m3, m4 = st.columns(4)
        base_daily = m1.number_input("Baseline daily revenue (LKR)", min_value=0.0, value=50000.0, step=1000.0)
        uplift_clear = m2.slider("Clear-day uplift (%)", 0, 50, 12)
        drop_rainy = m3.slider("Rainy-day drop (%)", 0, 50, 18)
        promo_cost = m4.number_input("Promo cost total (LKR)", min_value=0.0, value=15000.0, step=1000.0)

        rev_rows = []
        for _, r in df_calc.iterrows():
            factor = 1.0
            if r["is_clear"]: factor *= (1 + uplift_clear/100.0)
            if r["is_rainy"]: factor *= (1 - drop_rainy/100.0)
            rev = base_daily * factor
            rev_rows.append({"date": r["date"], "expected_revenue": round(rev, 0)})
        df_rev = pd.DataFrame(rev_rows)
        total_rev = float(df_rev["expected_revenue"].sum())
        roi = ((total_rev - 7*base_daily) - promo_cost) if promo_cost else (total_rev - 7*base_daily)

        st.dataframe(df_rev, use_container_width=True, hide_index=True)
        st.markdown(
            f"- **Projected 7-day revenue:** **LKR {total_rev:,.0f}**  \n"
            f"- **Δ vs baseline:** **LKR {(total_rev - 7*base_daily):,.0f}**  \n"
            f"- **ROI after promo cost:** **LKR {roi:,.0f}**"
        )
        st.download_button("⬇️ Download Revenue Projection (CSV)",
                           data=df_rev.to_csv(index=False).encode("utf-8"),
                           file_name="business_revenue_projection.csv", mime="text/csv",
                           use_container_width=True)

        # WhatsApp copy & Calendar
        wa_text = (
            f"Colombo weekly outlook:\n"
            f"- Best promo days: {', '.join(best_days) if best_days else '—'}\n"
            f"- Rainy days (≥10mm): {rainy_days}\n"
            f"- Hot days (≥30°C): {hot_days}\n"
            f"Plan: {(' • ').join(recs[:4]) if recs else '—'}"
        )
        st.text_area("📲 WhatsApp summary (copy & paste)", value=wa_text, height=120)

        ics_biz = _ics_for_days("Business: Promo focus day", best_days)
        st.download_button("🗓️ Add Promo Days to Calendar (.ics)", data=ics_biz.encode("utf-8"),
                           file_name="business_promo_days.ics", mime="text/calendar", use_container_width=True)

        export_business = {
            "weekly_stats": wk, "rainy_days": rainy_days, "hot_days": hot_days,
            "windy_days": windy_days, "clear_days": clear_days, "best_days": best_days,
            "biz_type": biz_type, "venue": venue, "priority": priority,
            "revenue_projection": rev_rows, "recommendations": recs
        }
        st.download_button(
            "⬇️ Download Business Plan (JSON)",
            data=json.dumps(export_business, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="colombo_business_plan.json",
            mime="application/json",
            use_container_width=True
        )

    # =========================
    # 🖨️ PROJECT PANEL / EXPORT
    # =========================
    with t_export:
        st.markdown("**Project Panel — Summary & Exports**")
        summary_md = [
            f"### Project: Weather-driven Decision Support (Colombo)",
            f"- Forecast horizon: **7 days**",
            f"- City: **{city_name or 'Colombo'}**",
            f"- Weekly rain: **{wk['rain_sum']:.0f} mm**, Avg temp: **{wk['tavg']:.1f}°C**, Max wind: **{wk['wind_max']:.0f} km/h**",
            f"- Risks → Water **{risks['water']}**, Heat **{risks['heat']}**, Wind **{risks['wind']}**, Disease **{risks['disease']}**",
            f"- Farmer sowing days: **{', '.join(sow_days) if sow_days else '—'}**",
            f"- Business best days: **{', '.join(best_days) if best_days else '—'}**",
        ]
        md_text = "\n".join(summary_md)
        st.markdown(md_text)

        st.download_button("⬇️ Download Project Summary (TXT)",
                           data=md_text.encode("utf-8"),
                           file_name="project_panel_summary.txt",
                           mime="text/plain",
                           use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# Footer
st.caption("Built with Streamlit • Plotly • TensorFlow | Live weather + map + icons ✨")
