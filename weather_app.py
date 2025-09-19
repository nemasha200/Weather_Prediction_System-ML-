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
  <a class="users"    href="?users=1">Contact</a>
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
st.title("🧭 Admin dashboard")
st.caption(
    "Signed-in experience for fast 7-day predictions + decision support for agriculture and local businesses in Colombo"
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



    

# Footer
st.caption("Built with Streamlit • Plotly • TensorFlow | Live weather + map + icons ✨")
