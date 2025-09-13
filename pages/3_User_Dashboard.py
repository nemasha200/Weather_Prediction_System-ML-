# pages/3_User_Dashboard.py — UPDATED with Decision Support (Farmers & Local Businesses)
# -*- coding: utf-8 -*-
import os, io, math, base64, numpy as np, pandas as pd, urllib.parse
from datetime import datetime, timedelta
import streamlit as st, plotly.express as px, requests
from tensorflow.keras.models import load_model
import joblib

# ──────────────────────────────────────────────────────────────────────────────
# Page & Auth Guard
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="User Dashboard • 7-Day Forecast", page_icon="🧭", layout="wide")

def _qp_get():
    try:
        return dict(st.query_params)  # Streamlit ≥1.30
    except Exception:
        return st.experimental_get_query_params()  # older

def _qp_set(d: dict):
    try:
        st.query_params = d        # Streamlit ≥1.30
    except Exception:
        st.experimental_set_query_params(**d)

qp = _qp_get()
if qp.get("logout") == "1":
    for k in list(st.session_state.keys()):
        if k.startswith("auth_") or k in ("auth_ok",):
            st.session_state.pop(k, None)
    _qp_set({})
    for target in ("pages/0_🔐_Login.py", "pages/0_Login.py", "🔐 Login", "login.py"):
        try:
            st.switch_page(target); st.stop()
        except Exception:
            pass
    st.stop()

if not st.session_state.get("auth_ok", False):
    for target in ("pages/0_🔐_Login.py", "pages/0_Login.py", "🔐 Login"):
        try:
            st.switch_page(target); st.stop()
        except Exception:
            pass
    st.warning("Please log in first.", icon="🔑")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Look & Feel
# ──────────────────────────────────────────────────────────────────────────────
def _b64_image(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

_bg64 = _b64_image("cloud2.jpg") or _b64_image(os.path.join("lstm", "cloud2.jpg"))
if _bg64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{_bg64}") no-repeat center center fixed;
            background-size: cover;
        }}
        header[data-testid="stHeader"] {{ background:#0b3d66 !important; border-bottom:1px solid rgba(255,255,255,.12); }}
        .pill {{ display:inline-block; padding:4px 10px; border-radius:999px; margin:2px; font-weight:700; font-size:12px; }}
        .pill.rain {{ background:#1e3a8a; color:#fff; }}
        .pill.wind {{ background:#155e75; color:#fff; }}
        .pill.heat {{ background:#b91c1c; color:#fff; }}
        .pill.uv {{ background:#7c3aed; color:#fff; }}
        .pill.ok {{ background:#065f46; color:#fff; }}
        .flagged {{ background:rgba(255,255,255,.7); border-radius:12px; padding:10px; }}
        .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:12px 0; }}
        .card {{ background:rgba(255,255,255,.9); border-radius:14px; padding:14px; box-shadow:0 4px 12px rgba(0,0,0,.1); }}
        .card h3 {{ margin:0 0 6px 0; font-size:16px; }}
        .card .big {{ font-size:18px; font-weight:800; }}
        .hi-badge {{ font-weight:700; padding:2px 8px; border-radius:999px; background:#0ea5e9; color:#fff; }}
        .conf {{ padding:6px 10px; border-radius:12px; display:inline-block; font-weight:700; }}
        .conf.high {{ background:#e6fffb; color:#0f766e; border:1px solid #99f6e4; }}
        .conf.med {{ background:#fffde7; color:#a16207; border:1px solid #fde68a; }}
        .conf.low {{ background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }}
        .whatif {{ background:rgba(255,255,255,.85); border-radius:12px; padding:12px; }}

        </style>
        """,
        unsafe_allow_html=True,
    )

# Sticky top header text + logout
st.markdown(
    """
<style>
header[data-testid="stHeader"]::before {
    content: "Weather Prediction System";
    font-size: 30px; font-weight: bold; color: white;
    position: absolute; left: 50%; transform: translateX(-50%);
}
.top-actions{position:fixed;top:70px;right:18px;z-index:9999;display:flex;gap:8px}
.top-actions a{padding:8px 12px;border-radius:8px;border:1px solid rgba(255,255,255,.25);
text-decoration:none;color:#fff !important;box-shadow:0 2px 6px rgba(0,0,0,.12);font-weight:700}
.top-actions a.logout{background:green} .top-actions a.logout:hover{background:black}
</style>
<div class="top-actions"><a class="logout" href="?logout=1">Logout</a></div>
""",
    unsafe_allow_html=True,
)

st.title("🧭 User dashboard")
st.caption(
    "Signed-in experience for fast 7-day predictions + decision support for agriculture and local businesses in Colombo"
)

st.markdown("""
<style>
/* Main content buttons (e.g., 'Predict Next 7 Days') */
div[data-testid="stAppViewContainer"] .stButton > button {
  background-color: #0796a3 !important;   /* light blue */
  border: 1px solid #93c5fd !important;
  color: #0f172a !important;               /* dark text for contrast */
  box-shadow: 0 2px 6px rgba(0,0,0,.08) !important;
}

/* Slightly darker on hover */
div[data-testid="stAppViewContainer"] .stButton > button:hover,
div[data-testid="stAppViewContainer"] .stButton > button:focus {
  background-color: #ba9fd6 !important;   /* light blue 400 */
  border-color: #60a5fa !important;
}
</style>
""", unsafe_allow_html=True)


# Sidebar look (UPDATED: keep labels white, force black text inside inputs)
st.markdown(
    """
<style>
/* Sidebar gradient background */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0F3D7A 0%, #0A2B5E 50%, #062048 100%) !important;
}
/* Default sidebar text (labels/headings) white */
section[data-testid="stSidebar"] * {
  color: #ffffff !important;
}

/* Force black text inside input/select/textarea controls */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stDateInput input,
section[data-testid="stSidebar"] .stTimeInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
/* BaseWeb select used by selectbox/multiselect */
section[data-testid="stSidebar"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] .stMultiSelect div[role="combobox"] * {
  color: #000 !important;
}

/* White backgrounds for form fields for contrast */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stDateInput input,
section[data-testid="stSidebar"] .stTimeInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background-color: #ffffff !important;
}

/* Placeholder color */
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
  color: #6b7280 !important; /* gray-500 */
  opacity: 1;
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Model IO
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "temp", "humidity", "precip", "windspeed", "winddir",
    "cloudcover", "dew", "uvindex", "sealevelpressure"
]
ALL_FEATURES = FEATURE_COLUMNS + [
    "day_of_year", "month", "day_of_week",
    "day_sin", "day_cos", "month_sin", "month_cos"
]

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    model_path_h5 = os.path.join("lstm", "lstm_model.h5")
    keras_path = os.path.join("lstm", "lstm_model.keras")
    scaler_path = os.path.join("lstm", "scaler.joblib")
    if not os.path.exists(scaler_path):
        if os.path.exists("scaler.joblib"):
            scaler_path = "scaler.joblib"
        else:
            raise FileNotFoundError("Scaler not found in ./lstm or current folder.")
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
    return model, scaler

def add_date_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])  # expect hourly or daily
    df = df.sort_values("datetime").reset_index(drop=True)
    for c in FEATURE_COLUMNS:
        if c in df.columns:
            df[c] = df[c].fillna(method="ffill").fillna(method="bfill")
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["day_sin"] = np.sin(2*np.pi*df["day_of_year"]/365)
    df["day_cos"] = np.cos(2*np.pi*df["day_of_year"]/365)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    return df

def build_today_row(temp, humidity, precip, windspeed, winddir, cloudcover, dew, uvindex, sealevelpressure, now=None):
    now = now or datetime.now()
    doy = now.timetuple().tm_yday
    month = now.month
    dow = now.weekday()
    day_sin = np.sin(2*np.pi*doy/365); day_cos = np.cos(2*np.pi*doy/365)
    month_sin = np.sin(2*np.pi*month/12); month_cos = np.cos(2*np.pi*month/12)
    return np.array([
        temp, humidity, precip, windspeed, winddir, cloudcover, dew, uvindex, sealevelpressure,
        doy, month, dow, day_sin, day_cos, month_sin, month_cos
    ]).reshape(1, -1)

def make_forecast(model, scaler, recent_block, base_date, horizon=7):
    X = recent_block.reshape(1, recent_block.shape[0], recent_block.shape[1])
    pred_scaled = model.predict(X).reshape(horizon, len(FEATURE_COLUMNS))
    dummy = np.zeros((horizon, scaler.n_features_in_))
    dummy[:, :len(FEATURE_COLUMNS)] = pred_scaled
    unscaled = scaler.inverse_transform(dummy)[:, :len(FEATURE_COLUMNS)]
    dates = [(base_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(horizon)]
    out = pd.DataFrame(unscaled, columns=FEATURE_COLUMNS)
    out.insert(0, "date", dates)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Derived metrics & helpers
# ──────────────────────────────────────────────────────────────────────────────
def heat_index_c(t_c, rh):
    t_f = t_c*9/5 + 32
    hi_f = -42.379 + 2.04901523*t_f + 10.14333127*rh - .22475541*t_f*rh \
           - .00683783*t_f*t_f - .05481717*rh*rh + .00122874*t_f*t_f*rh \
           + .00085282*t_f*rh*rh - .00000199*t_f*t_f*rh*rh
    return (hi_f-32)*5/9

def wet_bulb_stull(t_c, rh):
    rh = max(1e-3, min(100.0, rh))
    Tw = t_c * math.atan(0.151977 * (rh + 8.313659) ** 0.5) + \
         math.atan(t_c + rh) - math.atan(rh - 1.676331) + \
         0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh) - 4.686035
    return Tw

def rain_class(mm):
    if mm >= 50: return "torrential"
    if mm >= 20: return "heavy"
    if mm >= 10: return "moderate"
    if mm > 0:   return "light"
    return "none"

def uv_risk(uvi):
    if uvi >= 11: return "extreme"
    if uvi >= 8:  return "very high"
    if uvi >= 6:  return "high"
    if uvi >= 3:  return "moderate"
    return "low"

def wind_risk(kmh):
    if kmh >= 60: return "gale"
    if kmh >= 40: return "strong"
    if kmh >= 25: return "breezy"
    return "calm/normal"

def drying_score(temp_c, rh, wind_kmh, rain_mm):
    # Simple composite for laundry/paint/concrete drying potential
    score = 0.8*(temp_c - 20) + 0.5*(80 - rh) + 0.4*wind_kmh
    if rain_mm > 0: score -= 40
    return float(score)

def is_rainsafe(pop_pct_like, rain_mm, wind_kmh, pop_thresh=30, rain_thresh=0.5, wind_thresh=35):
    proxy_pop = 70 if rain_mm >= 1.0 else 20  # coarse proxy
    p = pop_pct_like if pop_pct_like is not None else proxy_pop
    return (p < pop_thresh) and (rain_mm < rain_thresh) and (wind_kmh < wind_thresh)

def field_workability(last48_rain_mm, next24_rain_mm, et0_next24_mm, thresholds=(20, 10, 4)):
    ok = (last48_rain_mm <= thresholds[0]) and (next24_rain_mm <= thresholds[1]) and (et0_next24_mm >= thresholds[2])
    score = (thresholds[0]-last48_rain_mm) + (thresholds[1]-next24_rain_mm) + (et0_next24_mm-thresholds[2])*5
    return ok, float(score)

def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["heat_index"] = [heat_index_c(t, rh) for t, rh in zip(df["temp"], df["humidity"])]
    df["wet_bulb"]   = [wet_bulb_stull(t, rh) for t, rh in zip(df["temp"], df["humidity"])]
    df["rain_class"] = df["precip"].apply(rain_class)
    df["uv_risk"]    = df["uvindex"].apply(uv_risk)
    df["wind_risk"]  = df["windspeed"].apply(wind_risk)
    df["dry_score"]  = [drying_score(t, rh, w, r) for t, rh, w, r in zip(df["temp"], df["humidity"], df["windspeed"], df["precip"])]
    return df

def climatology_confidence(df: pd.DataFrame) -> str:
    # Very rough heuristic ranges for Colombo (daily): temp 24–33, RH 60–95, rain 0–60
    t_ok = ((df["temp"] >= 24) & (df["temp"] <= 33)).mean()
    rh_ok = ((df["humidity"] >= 60) & (df["humidity"] <= 95)).mean()
    r_ok = ((df["precip"] >= 0) & (df["precip"] <= 60)).mean()
    score = (t_ok + rh_ok + r_ok) / 3
    if score > 0.8: return "high"
    if score > 0.55: return "med"
    return "low"

# ──────────────────────────────────────────────────────────────────────────────
# Localization (EN / SI)
# ──────────────────────────────────────────────────────────────────────────────
LANG = st.sidebar.selectbox("Language / භාෂාව", ["English", "සිංහල"], index=0)
def tr(en, si):
    return si if LANG == "සිංහල" else en

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — model + input mode + live
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ " + tr("Forecast Controls", "කලින් අනාවැකි පාලක"))
    try:
        model, scaler = load_model_and_scaler()
        st.success(tr("Model ready", "මොඩලය සූදානම්"), icon="✅")
    except Exception as e:
        st.error(f"Model/scaler load failed: {e}")
        st.stop()

    mode = st.radio(
        tr("Input Mode", "ආදාන ක්‍රමය"),
        [tr("Enter today's values", "අද දත්ත ඇතුල් කරන්න"), tr("Upload last 30 days (CSV)", "දින 30ක CSV දමන්න")],
        index=0,
    )

    st.markdown("### 🌍 " + tr("(Optional) Live Weather", "(විකල්ප) සජීව කාලගුණය"))
    api_key = st.text_input(tr("OpenWeatherMap API key", "OpenWeatherMap යතුර"), type="password")
    city = st.text_input(tr("City (e.g., Colombo)", "නගරය (කොළඹ වගේ)"))
    fetch_btn = st.button("📡 " + tr("Use current city weather", "දැනට ඇති නගරයේ දත්ත භාවිතා කරන්න"))

# Defaults
_default_values = dict(
    temp=28.0, humidity=75.0, precip=2.0, windspeed=12.0, winddir=180.0,
    cloudcover=50.0, dew=22.0, uvindex=7.0, sealevelpressure=1010.0,
)
for k, v in _default_values.items():
    st.session_state.setdefault(k, v)

# Threshold defaults (persistable)
st.session_state.setdefault("rain_thr", 10.0)
st.session_state.setdefault("wind_thr", 35.0)
st.session_state.setdefault("heat_thr", 32.0)
st.session_state.setdefault("uv_thr", 8.0)

# Live fetch
if fetch_btn:
    if not api_key or not city:
        st.sidebar.error(tr("Enter API key and City.", "යතුර සහ නගරය දෙකම දෙන්න."))
    else:
        try:
            r = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city, "appid": api_key, "units": "metric"},
                timeout=10,
            )
            r.raise_for_status()
            d = r.json()
            main, wind, clouds, rain, snow = (
                d.get("main", {}), d.get("wind", {}), d.get("clouds", {}), d.get("rain", {}), d.get("snow", {})
            )
            temp = float(main.get("temp", st.session_state["temp"]))
            humidity = float(main.get("humidity", st.session_state["humidity"]))
            pressure = float(main.get("pressure", st.session_state["sealevelpressure"]))
            wind_kmh = float(wind.get("speed", 0.0)) * 3.6
            wind_dir = float(wind.get("deg", st.session_state["winddir"]))
            cloud_pct = float(clouds.get("all", st.session_state["cloudcover"]))
            precip = float(rain.get("1h", 0.0) or snow.get("1h", 0.0) or 0.0)
            uv = st.session_state["uvindex"]
            coord = d.get("coord", {})
            if coord:
                try:
                    r2 = requests.get(
                        "https://api.openweathermap.org/data/3.0/onecall",
                        params={
                            "lat": coord.get("lat"),
                            "lon": coord.get("lon"),
                            "appid": api_key,
                            "units": "metric",
                            "exclude": "minutely,hourly,daily,alerts",
                        },
                        timeout=10,
                    )
                    if r2.ok:
                        uv = float(r2.json().get("current", {}).get("uvi", uv))
                except Exception:
                    pass
            # Dewpoint (Magnus)
            a, b = 17.62, 243.12
            gamma = (a*temp)/(b+temp) + math.log(max(min(humidity,100.0),1e-2)/100.0)
            dew = (b*gamma)/(a-gamma)

            st.session_state.update(
                dict(
                    temp=round(temp,1), humidity=round(humidity,0), precip=round(precip,1),
                    windspeed=round(wind_kmh,1), winddir=round(wind_dir,0), cloudcover=round(cloud_pct,0),
                    dew=round(dew,1), uvindex=round(uv,1), sealevelpressure=round(pressure,0),
                    city_name=d.get("name", city),
                )
            )
            st.sidebar.success(tr("Live weather applied ✅", "සජීව දත්ත යාවත්කාලීන ✅"))
            st.rerun()
        except Exception as ex:
            st.sidebar.error(f"Fetch failed: {ex}")

# ──────────────────────────────────────────────────────────────────────────────
# Inputs & Predict Button (computation only)
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.subheader("📥 " + tr("Provide Input", "දත්ත ලබාදෙන්න"))
    st.info(
        tr("Either enter today's values or upload a CSV with the last 30 days.", "අද දත්ත ඇතුල් කරන්න හෝ දින 30ක CSV එකක් උඩුගත කරන්න."),
        icon="ℹ️",
    )

    history_df = None
    if mode.startswith("Upload") or mode.startswith("දින"):
        st.write(tr("CSV needs `datetime` + ", "`datetime` + ") + ", ".join(f"`{c}`" for c in FEATURE_COLUMNS))
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            try:
                raw = up.read()
                history_df = pd.read_csv(io.BytesIO(raw))
                if "datetime" not in history_df.columns:
                    st.error(tr("CSV must contain `datetime` column.", "`datetime` තීරුව අවශ්‍යයි."))
                    history_df = None
                else:
                    history_df = add_date_features(history_df)
                    missing = [c for c in FEATURE_COLUMNS if c not in history_df.columns]
                    if missing:
                        st.error(tr(f"Missing: {missing}", f" හිඟයි: {missing}"))
                        history_df = None
                    else:
                        st.success(tr("CSV loaded.", "CSV එක OK."), icon="📄")
            except Exception as e:
                st.error(f"Parse error: {e}")
                history_df = None
    else:
        c1, c2, c3 = st.columns(3)
        st.session_state["temp"] = c1.number_input(tr("Temp (°C)", "උෂ්ණත්වය (°C)"), value=float(st.session_state["temp"]))
        st.session_state["humidity"] = c2.number_input(
            tr("Humidity (%)", "සාපේක්ෂ ආර්ද්‍රතාව (%)"), value=float(st.session_state["humidity"]), min_value=0.0, max_value=100.0
        )
        st.session_state["precip"] = c3.number_input(
            tr("Precip (mm)", "වර්ෂාප්‍රමාණය (මි.මී.)"), value=float(st.session_state["precip"]), min_value=0.0
        )

        c4, c5, c6 = st.columns(3)
        st.session_state["windspeed"] = c4.number_input(
            tr("Wind Speed (km/h)", "සුළඟ වේගය (km/h)"), value=float(st.session_state["windspeed"]), min_value=0.0
        )
        st.session_state["winddir"] = c5.number_input(
            tr("Wind Dir (°)", "සුළඟ දිශාව (°)"), value=float(st.session_state["winddir"]), min_value=0.0, max_value=360.0
        )
        st.session_state["cloudcover"] = c6.number_input(
            tr("Cloud Cover (%)", "මේඛලා ආවරණය (%)"), value=float(st.session_state["cloudcover"]), min_value=0.0, max_value=100.0
        )

        c7, c8, c9 = st.columns(3)
        st.session_state["dew"] = c7.number_input(tr("Dew Point (°C)", "දාහ බින්දු (°C)"), value=float(st.session_state["dew"]))
        st.session_state["uvindex"] = c8.number_input(tr("UV Index", "UV දර්ශකය"), value=float(st.session_state["uvindex"]), min_value=0.0)
        st.session_state["sealevelpressure"] = c9.number_input(
            tr("Sea Level Pressure (hPa)", "පීඩනය (hPa)"), value=float(st.session_state["sealevelpressure"])
        )

with right:
    st.subheader("🔮 " + tr("Forecast", "අනාවැකිය"))
    go = st.button("🚀 " + tr("Predict Next 7 Days", "දින 7ක් අනාවැකිය"), type="primary", use_container_width=True)


# NEW: Quick Insights (Today) under the button
    try:
        t = float(st.session_state.get("temp", 28.0))
        rh = float(st.session_state.get("humidity", 75.0))
        r  = float(st.session_state.get("precip", 0.0))
        w  = float(st.session_state.get("windspeed", 10.0))
        uv = float(st.session_state.get("uvindex", 7.0))

        def drying_score_local(temp_c, rh, wind_kmh, rain_mm):
            s = 0.8*(temp_c - 20) + 0.5*(80 - rh) + 0.4*wind_kmh
            if rain_mm > 0: s -= 40
            return float(s)

        ds = drying_score_local(t, rh, w, r)
        ds_tag = "Great" if ds >= 30 else ("Fair" if ds >= 10 else "Poor")
        hi = heat_index_c(t, rh)
        hi_tag = tr("Very High","වලංගු ඉහළ") if hi >= 41 else (tr("High","ඉහළ") if hi >= 32 else (tr("Moderate","මධ්‍ය") if hi >= 27 else tr("Low","අඩු")))
        uv_tag = uv_risk(uv)
        safe = is_rainsafe(None, r, w)

        qi_html = f"""
        <div class='card' style="margin-top:8px;">
          <h3>✨ {tr('Quick Insights (Today)', 'ඉක්මන් අඳහස් (අද)')}</h3>
          <div class='cards' style="grid-template-columns:repeat(4,minmax(0,1fr));">
            <div class='card'>
              <h3>🧺 {tr('Drying', 'වියළීම')}</h3>
              <div class='big'>{ds_tag}</div>
              <div>{tr('Score','ස්කෝර්')}: {ds:.0f}</div>
              <small>{tr('Laundry/paint/concrete potential','වස්ත්‍ර/පේන්ට්/බෙටන් වියළීම')}</small>
            </div>
            <div class='card'>
              <h3>🥵 {tr('Heat Index','උණ දර්ශකය')}</h3>
              <div class='big'>{hi:.1f}°C</div>
              <div>{tr('Risk','අපදාව')}: {hi_tag}</div>
              <small>{tr('Hydrate; shade at midday','වතුර බොන්න; මධ්‍යහ්න ශේඩ්')}</small>
            </div>
            <div class='card'>
              <h3>🌞 UV</h3>
              <div class='big'>{uv:.1f}</div>
              <div>{tr('Risk','අපදාව')}: {uv_tag.title()}</div>
              <small>{tr('Prefer early/late outdoor work','අළුයම්/සවස වැඩ කරන්න')}</small>
            </div>
            <div class='card'>
              <h3>🌂 {tr('Rain/Wind Safety','වැසි/සුළඟ ආරක්ෂාව')}</h3>
              <div class='big'>{tr('Safe','ආරක්ෂිත') if safe else tr('Not Ideal','සුදුසු නැහැ')}</div>
              <div>{tr('Rain','වැසි')}: {r:.1f} mm · {tr('Wind','සුළඟ')}: {w:.0f} km/h</div>
              <small>{tr('Carry cover / adjust plans','ආවරණ ගන්න / සැලැස්ම වෙනස් කරන්න')}</small>
            </div>
          </div>
        </div>
        """
        st.markdown(qi_html, unsafe_allow_html=True)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Forecast computation (runs only when button clicked)
# ──────────────────────────────────────────────────────────────────────────────
if go:
    try:
        if (mode.startswith("Upload") or mode.startswith("දින")) and history_df is not None:
            base_date = pd.to_datetime(history_df["datetime"].iloc[-1])
            block = history_df.tail(30).copy()
            while len(block) < 30:
                block = pd.concat([history_df.tail(1), block], ignore_index=True)
            recent = block[ALL_FEATURES].values
            recent_scaled = scaler.transform(recent)
            forecast_df = make_forecast(model, scaler, recent_scaled, base_date, 7)
        else:
            today = build_today_row(
                st.session_state["temp"], st.session_state["humidity"], st.session_state["precip"],
                st.session_state["windspeed"], st.session_state["winddir"], st.session_state["cloudcover"],
                st.session_state["dew"], st.session_state["uvindex"], st.session_state["sealevelpressure"],
            )
            recent_scaled = scaler.transform(today)
            recent_scaled_30 = np.repeat(recent_scaled, 30, axis=0)
            forecast_df = make_forecast(model, scaler, recent_scaled_30, datetime.now(), 7)

        forecast_df = add_derived(forecast_df)

        # persist results for future reruns
        st.session_state["user_forecast_df"] = forecast_df.copy()
        st.session_state["has_forecast"] = True

        st.success(tr("Forecast computed ✅", "අනාවැකිය සම්පූර්ණයි ✅"))
        # Optional: st.rerun()  # not necessary; we render below when flag is set
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Rendering (always runs when we have forecast in session)
# ──────────────────────────────────────────────────────────────────────────────
def render_forecast_and_decisions():
    forecast_df = st.session_state["user_forecast_df"].copy()

    # Quick KPIs (Day 1)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(tr("Day 1 Temp (°C)", "දින 1 උෂ්ණත්වය (°C)"), f"{forecast_df.loc[0,'temp']:.1f}")
    k2.metric(tr("Day 1 RH (%)", "දින 1 RH (%)"), f"{forecast_df.loc[0,'humidity']:.0f}")
    k3.metric(tr("Day 1 Rain (mm)", "දින 1 වැසි (මි.මී.)"), f"{forecast_df.loc[0,'precip']:.1f}")
    k4.metric(tr("Day 1 Wind (km/h)", "දින 1 සුළඟ (km/h)"), f"{forecast_df.loc[0,'windspeed']:.0f}")

    # Threshold/alerts controls (persist)
    st.markdown("### 🚨 " + tr("Alerts & Risk Thresholds", "අභිලාෂයන් / අවදානම්"))
    cA, cB, cC, cD = st.columns(4)
    cA.number_input(tr("Rainy ≥ (mm)", "වැසි ≥ (මි.මී.)"),
                    value=float(st.session_state.get("rain_thr", 10.0)), min_value=0.0, key="rain_thr")
    cB.number_input(tr("Windy ≥ (km/h)", "සුළඟ ≥ (km/h)"),
                    value=float(st.session_state.get("wind_thr", 35.0)), min_value=0.0, key="wind_thr")
    cC.number_input(tr("Heat Index ≥ (°C)", "දහන දර්ශකය ≥ (°C)"),
                    value=float(st.session_state.get("heat_thr", 32.0)), min_value=0.0, key="heat_thr")
    cD.number_input(tr("High UV ≥", "අධික UV ≥"),
                    value=float(st.session_state.get("uv_thr", 8.0)), min_value=0.0, key="uv_thr")

    fx = forecast_df.copy()
    fx["is_rainy"] = fx["precip"]   >= st.session_state["rain_thr"]
    fx["is_windy"] = fx["windspeed"]>= st.session_state["wind_thr"]
    fx["is_hot"]   = fx["heat_index"]>= st.session_state["heat_thr"]
    fx["is_uv"]    = fx["uvindex"]  >= st.session_state["uv_thr"]

    # keep for other tabs
    st.session_state["fx_for_decisions"] = fx.copy()

    # Confidence indicator
    conf = climatology_confidence(fx)
    conf_html = {
        "high": f"<span class='conf high'>{tr('Confidence: HIGH','විශ්වාසය: ඉහළ')}</span>",
        "med":  f"<span class='conf med'>{tr('Confidence: MED','විශ්වාසය: මධ්‍ය')}</span>",
        "low":  f"<span class='conf low'>{tr('Confidence: LOW','විශ්වාසය: අඩු')}</span>",
    }[conf]
    st.markdown(conf_html, unsafe_allow_html=True)

    # Flagged days strip
    flagged = []
    for _, r in fx.iterrows():
        pills = []
        if r["is_rainy"]: pills.append(f'<span class="pill rain">{tr("Rain","වැසි")}</span>')
        if r["is_windy"]: pills.append(f'<span class="pill wind">{tr("Wind","සුළඟ")}</span>')
        if r["is_hot"]:   pills.append(f'<span class="pill heat">{tr("Heat","උණ")}</span>')
        if r["is_uv"]:    pills.append(f'<span class="pill uv">{tr("UV","UV")}</span>')
        if not pills:      pills.append(f'<span class="pill ok">{tr("OK","හරි")}</span>')
        flagged.append(f"<div><b>{r['date']}</b> → {' '.join(pills)}</div>")
    st.markdown('<div class="flagged">' + "".join(flagged) + "</div>", unsafe_allow_html=True)

    # WHAT-IF sandbox
    with st.expander("🧪 " + tr("What-if sandbox (tweak forecast to test decisions)", "What-if Sandbox (දත්ත ට්වික් කරන්න)")):
        st.markdown("<div class='whatif'>", unsafe_allow_html=True)
        w1, w2, w3, w4 = st.columns(4)
        dT = w1.slider(tr("Δ Temp (°C)", "Δ උෂ්ණත්වය (°C)"), -3.0, 3.0, 0.0, 0.5)
        dRH = w2.slider(tr("Δ RH (%)", "Δ RH (%)"), -15, 15, 0, 1)
        dWind = w3.slider(tr("Δ Wind (km/h)", "Δ සුළඟ (km/h)"), -10, 10, 0, 1)
        rain_factor = w4.slider(tr("× Rain (scale)", "× වැසි (පමානය)"), 0.5, 1.5, 1.0, 0.05)
        st.markdown("</div>", unsafe_allow_html=True)

        fx_adj = fx.copy()
        fx_adj["temp"] += dT
        fx_adj["humidity"] = np.clip(fx_adj["humidity"] + dRH, 0, 100)
        fx_adj["windspeed"] = np.clip(fx_adj["windspeed"] + dWind, 0, None)
        fx_adj["precip"] = np.clip(fx_adj["precip"] * rain_factor, 0, None)
        fx_adj = add_derived(fx_adj)
        fx_adj["is_rainy"] = fx_adj["precip"] >= st.session_state["rain_thr"]
        fx_adj["is_windy"] = fx_adj["windspeed"] >= st.session_state["wind_thr"]
        fx_adj["is_hot"]   = fx_adj["heat_index"] >= st.session_state["heat_thr"]
        fx_adj["is_uv"]    = fx_adj["uvindex"] >= st.session_state["uv_thr"]
        st.session_state["fx_for_decisions"] = fx_adj

    dfx = st.session_state["fx_for_decisions"]; dfc = dfx.copy()
    # Best drying day
    best_dry = dfc.sort_values("dry_score", ascending=False).iloc[0]
    # Lowest rain + low wind day
    dfc["r_w_score"] = (dfc["precip"].rank(ascending=True) + dfc["windspeed"].rank(ascending=True))
    best_ops = dfc.sort_values("r_w_score").iloc[0]
    # Peak heat & peak wind day
    peak_heat = dfc.sort_values("heat_index", ascending=False).iloc[0]
    peak_wind = dfc.sort_values("windspeed", ascending=False).iloc[0]

    cards_html = f"""
        <div class='cards'>
          <div class='card'>
            <h3>🧺 {tr('Best Drying','වියළීම හොඳම')}</h3>
            <div class='big'>{best_dry['date']}</div>
            <div>{tr('Drying score','වියළීම ස್ಕෝර්')}: {best_dry['dry_score']:.0f}</div>
          </div>
          <div class='card'>
            <h3>🛠️ {tr('Best Outdoor Ops','වෙළෙඳ/ක්ෂේත්‍ර වැඩට හොඳම')}</h3>
            <div class='big'>{best_ops['date']}</div>
            <div>{tr('Rain','වැසි')}: {best_ops['precip']:.1f} mm · {tr('Wind','සුළඟ')}: {best_ops['windspeed']:.0f} km/h</div>
          </div>
          <div class='card'>
            <h3>🥵 {tr('Heat Risk Peak','උණ අවදානම ඉහළම')}</h3>
            <div class='big'>{peak_heat['date']}</div>
            <div>HI {peak_heat['heat_index']:.1f}°C · RH {peak_heat['humidity']:.0f}%</div>
          </div>
          <div class='card'>
            <h3>🌬️ {tr('Wind Alert','සුළඟ අවවාදය')}</h3>
            <div class='big'>{peak_wind['date']}</div>
            <div>{tr('Max wind','උපරිම සුළඟ')}: {peak_wind['windspeed']:.0f} km/h</div>
          </div>
        </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

    # Tabs
    t_over, t_agri, t_biz, t_share = st.tabs([
        tr("📈 Overview", "📈 සැලැස්ම"),
        tr("🌾 Agriculture", "🌾 කෘෂිකර්මය"),
        tr("🏪 Local Business", "🏪 දේශීය ව්‍යාපාර"),
        tr("🔗 Share/Export", "🔗 බෙදාගැනීම්/නිර්යාත"),
    ])

    with t_over:
        template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
        plot_df = forecast_df.melt(
            id_vars="date",
            value_vars=["temp", "humidity", "precip", "windspeed", "uvindex"],
            var_name="feature",
            value_name="value",
        )
        fig = px.line(
            plot_df,
            x="date",
            y="value",
            color="feature",
            markers=True,
            title=tr("Key Weather Features (7 days)", "ප්‍රධාන කාලගුණ ගුණාංග (දින 7)"),
            template=template,
        )
        fig.update_layout(hovermode="x unified", legend_title_text=tr("Feature", "ලක්ෂණ"))
        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.dataframe(forecast_df.round(2), use_container_width=True, hide_index=True)

        st.markdown("**" + tr("🧾 Day-by-day summary:", "🧾 දින අනුව සාරාංශය:") + "**")
        for _, r in forecast_df.iterrows():
            desc = (
                f"{r['date']}: {r['temp']:.1f}°C (HI {r['heat_index']:.1f}°C), "
                f"RH {r['humidity']:.0f}%, rain {r['precip']:.1f} mm ({r['rain_class']}), "
                f"wind {r['windspeed']:.0f} km/h ({r['wind_risk']}), UV {r['uvindex']:.1f} ({r['uv_risk']})"
            )
            st.write("• " + desc)

        st.download_button(
            "⬇️ " + tr("Download Forecast (CSV)", "CSV ලෙස බාගන්න"),
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="user_forecast_7day.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Agriculture: Decision Assistant
    # ──────────────────────────────────────────────────────────────────────────
    with t_agri:
        st.markdown("### 🧠 " + tr("Decision Assistant (Fields)", "තීරණ උපකාරක (වැලිපිටි)"))
        col1, col2, col3 = st.columns(3)
        crop = col1.selectbox(tr("Crop", "බෝගය"), [
            "Paddy (Rice)", "Chilli", "Brinjal", "Okra", "Tomato", "Leafy Greens", "Coconut", "Tea",
        ])
        stage = col2.selectbox(tr("Growth Stage", "වැඩුන තත්ත්වය"), [
            "Nursery/Seedling", "Vegetative", "Flowering", "Fruiting", "Pre-Harvest",
        ])
        soil = col3.selectbox(tr("Soil/Bed", "මෘද/හැඩය"), ["Clayey", "Loam", "Sandy"], index=1)

        df = st.session_state["fx_for_decisions"].copy()
        total_rain = float(df["precip"].clip(lower=0).sum())
        tavg = float(df["temp"].mean())
        wind_max = float(df["windspeed"].max())
        uv_max = float(df["uvindex"].max())

        # irrigation need baseline (mm/week)
        base_need = {
            "Paddy (Rice)": 45, "Chilli": 30, "Brinjal": 30, "Okra": 28, "Tomato": 35,
            "Leafy Greens": 25, "Coconut": 40, "Tea": 35,
        }.get(crop, 30)

        soil_mod = {"Clayey": 0.9, "Loam": 1.0, "Sandy": 1.2}.get(soil, 1.0)
        stg_mod = {
            "Nursery/Seedling": 0.8, "Vegetative": 1.0, "Flowering": 1.2, "Fruiting": 1.1, "Pre-Harvest": 0.9,
        }.get(stage, 1.0)
        weekly_need = base_need * soil_mod * stg_mod
        net_irrig = max(0.0, weekly_need - total_rain)

        # Best sowing/transplant windows (low rain + comfy temp)
        tmp = df.copy()
        rain_norm = 1 - (tmp["precip"] / max(tmp["precip"].max(), 1e-6)).clip(0, 1)
        temp_norm = 1 - (abs(tmp["temp"] - 28) / 10).clip(0, 1)
        wind_norm = 1 - (tmp["windspeed"] / max(tmp["windspeed"].max(), 1e-6)).clip(0, 1)
        tmp["score"] = 0.55 * rain_norm + 0.30 * temp_norm + 0.15 * wind_norm
        best_sow = tmp.nlargest(2, "score")["date"].tolist()

        # Spray window (low rain, low wind, UV not extreme)
        spray = df[(df["precip"] < 2.0) & (df["windspeed"] < 15) & (df["uvindex"] <= 8)]["date"].tolist()[:3]

        # Fertilizer timing → avoid heavy rain days; choose next 2 moderate days
        fert = df[(df["precip"] < 8.0) & (df["windspeed"] < 25)].copy()
        fert["score"] = (1 - (abs(fert["temp"] - 27) / 8).clip(0, 1)) + (
            (fert["precip"].max() - fert["precip"]) / max(fert["precip"].max(), 1e-6)
        )
        fert_days = fert.nlargest(2, "score")["date"].tolist()

        # Harvest window: low rain + not too windy
        harvest = df[(df["precip"] < 5.0) & (df["windspeed"] < 25)]["date"].tolist()[:2]

        notes = []
        if net_irrig > 0:
            notes += [
                tr(
                    f"Irrigation need ~{net_irrig:.0f} mm this week (after rain).",
                    "මෙම සතියේ ජලසേചනය අවශ්‍ය ~{:.0f} මි.මී.".format(net_irrig),
                )
            ]
        if tavg >= 30:
            notes += [
                tr(
                    "Heat stress likely — shade cloth 30–40% for nurseries & midday mulching.",
                    "උෂ්ණතා ආබාධය Sambhavana — නර්සරී වල 30–40% ශේඩ්, මධ්‍යහ්න කාලයේ මල්චින්ග්.",
                )
            ]
        if uv_max >= 8:
            notes += [
                tr(
                    "UV very high on some days — schedule field work early/late; PPE essential.",
                    "හෙළිදා නොයෙක් දින UV ඉහළයි — අළුයම/සවස වැඩ සැලසුම් කර PPE පාවිච්චි කරන්න.",
                )
            ]
        if wind_max >= 35:
            notes += [
                tr("Gusty day expected — stake vines/okra; secure covers.", "බලවත් සුළඟ — වැල්/බණ්ඩාකාර බෝග අස්සේ සවි කරන්න; ආවරණ තදකරන්න."),
            ]

        st.markdown("**" + tr("This Week’s Field Plan", "මෙම සතියේ ක්ෂේත්‍ර සැලැස්ම") + ":**")
        st.write(
            "• "
            + tr(
                f"Sowing/Transplant: {', '.join(best_sow) if best_sow else '—'}",
                f"වපුරීම/ප්‍රතිරෝපණය: {', '.join(best_sow) if best_sow else '—'}",
            )
        )
        st.write(
            "• "
            + tr(
                f"Spray Window (low wind/rain): {', '.join(spray) if spray else '—'}",
                f"පසි කිරීම් කාලය (කුඩා සුළඟ/වැසි): {', '.join(spray) if spray else '—'}",
            )
        )
        st.write(
            "• "
            + tr(
                f"Fertilizer Days: {', '.join(fert_days) if fert_days else '—'}",
                f"පෝෂක දීමේ දින: {', '.join(fert_days) if fert_days else '—'}",
            )
        )
        st.write(
            "• "
            + tr(
                f"Harvest Window: {', '.join(harvest) if harvest else '—'}",
                f"අස්වනු රැස්කරගැනීම: {', '.join(harvest) if harvest else '—'}",
            )
        )

        if notes:
            st.markdown("**" + tr("Notes / Cautions", "සටහන් / අවවාද") + ":**")
            st.write("• " + "\n• ".join(notes))

        # Export field plan
        plan_lines = [
            f"Crop: {crop} | Stage: {stage} | Soil: {soil}",
            f"Weekly rain (sum): {total_rain:.0f} mm | Mean temp: {tavg:.1f} °C",
            f"Irrigation need (net): ~{net_irrig:.0f} mm",
            f"Sowing/Transplant: {', '.join(best_sow) if best_sow else '-'}",
            f"Spray Window: {', '.join(spray) if spray else '-'}",
            f"Fertilizer Days: {', '.join(fert_days) if fert_days else '-'}",
            f"Harvest Window: {', '.join(harvest) if harvest else '-'}",
        ] + (["Notes:"] + [f"- {n}" for n in notes] if notes else [])
        plan_text = "\n".join(plan_lines)
        st.download_button(
            "📝 " + tr("Download Weekly Field Plan (.txt)", "සතියේ සැලැස්ම බාගන්න (.txt)"),
            data=plan_text.encode("utf-8"),
            file_name="field_plan_week.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Local Business: Decision Assistant
    # ──────────────────────────────────────────────────────────────────────────
    with t_biz:
        st.markdown("### 🧠 " + tr("Decision Assistant (Operations)", "තීරණ උපකාරක (ක්‍රියාකාරී)"))
        b1, b2 = st.columns(2)
        biz = b1.selectbox(
            tr("Business Type", "ව්‍යාපාරය"),
            [
                "Retail Shop",
                "Restaurant/Café",
                "Outdoor Market/Events",
                "Delivery-First",
                "Laundry/Dry-Clean",
                "Construction",
            ],
        )
        open_days = b2.multiselect(
            tr("Days Open", "විවෘත දවස්"), forecast_df["date"].tolist(), default=forecast_df["date"].tolist()
        )

        df = st.session_state["fx_for_decisions"].copy()
        df["is_open"] = df["date"].isin(open_days)
        df["is_rainy"] = df["precip"] >= st.session_state["rain_thr"]
        df["is_clear"] = (df["cloudcover"] <= 30) & (~df["is_rainy"]).astype(bool)
        df["is_hot"] = df["heat_index"] >= st.session_state["heat_thr"]

        tmp = df[df["is_open"]].copy()
        if not tmp.empty:
            # Promo day scoring: prefer clearer, comfortable temp
            tmp["rain_score"] = (tmp["precip"].max() - tmp["precip"]) / max(tmp["precip"].max(), 1e-6)
            tmp["temp_score"] = 1 - (abs(tmp["temp"] - 28) / 10).clip(0, 1)
            tmp["footfall_score"] = 0.65 * tmp["rain_score"] + 0.35 * tmp["temp_score"]
            best_promo = tmp.nlargest(2, "footfall_score")["date"].tolist()
        else:
            best_promo = []

        rainy_days = int((df["is_open"] & df["is_rainy"]).sum())
        clear_days = int((df["is_open"] & df["is_clear"]).sum())
        hot_days = int((df["is_open"] & df["is_hot"]).sum())

        bullets = []
        if biz == "Retail Shop":
            if best_promo:
                bullets += [
                    tr(
                        f"In-store promo events on: {', '.join(best_promo)}",
                        f"ඇතුලත ප්‍රවර්ධන දවස්: {', '.join(best_promo)}",
                    )
                ]
            if rainy_days >= 2:
                bullets += [
                    tr(
                        "Push same-day delivery & WhatsApp orders on rainy days.",
                        "වැසි දවස්වල දිනයේම බෙදාහැරීම/WhatsApp ඇණවුම් ප්‍රවර්ධනය කරන්න.",
                    )
                ]
            if hot_days >= 2:
                bullets += [
                    tr("Stock cold drinks/ice; check A/C & backup power.", "සිසිල් පාන/අයිස්; A/C සහ විදුලි උපස්ථය පරීක්ෂා කරන්න."),
                ]
        elif biz == "Restaurant/Café":
            bullets += [
                tr(
                    "Staff up on clear/promo days; prep cold menu for heat days.",
                    "පසුම්බි දවස්වල වැඩි සේවකයන්; උණ දින වල සිසිල් අයිටම් වැඩි කරන්න.",
                )
            ]
            if rainy_days >= 2:
                bullets += [
                    tr(
                        "Run free delivery/‘rainy day’ combos; entry mats to avoid slips.",
                        "නොමිලේ බෙදාහැරීම/‘වැසි දවස’ කම්බෝ; පිවිසුම් මැට් තබන්න.",
                    )
                ]
        elif biz == "Outdoor Market/Events":
            bullets += [
                tr("Prefer low-rain & low-wind days for events; tents & anchors ready.", "කුඩා වැසි/සුළඟ දවස්වල අවස්ථා; ටෙන්ට්/ඇංකර් සූදානම්."),
            ]
        elif biz == "Delivery-First":
            bullets += [
                tr("Expect order spikes on rainy hot evenings; route & fuel planning.", "වැසි සහිත උණ සවසවල ඇණවුම් වැඩිවීම; මාර්ග/අන්ධන සැලැස්ම."),
            ]
        elif biz == "Laundry/Dry-Clean":
            best_drying_days = df.sort_values("dry_score", ascending=False)["date"].tolist()[:2]
            bullets += [
                tr(
                    f"Advertise pickup on rainy days; best drying: {', '.join(best_drying_days)}",
                    f"වැසි දවස්වල ගෙදරින් අරගැනීම ප්‍රවර්ධනය කරන්න; හොඳ වියළීම: {', '.join(best_drying_days)}",
                )
            ]
        elif biz == "Construction":
            concrete_ok = df[(df["precip"] < 2.0) & (df["humidity"] <= 85) & (df["windspeed"] < 35)]["date"].tolist()[:2]
            windy_warn = df[df["windspeed"] >= 40]["date"].tolist()
            bullets += [
                tr(
                    f"Concrete pour suitability: {', '.join(concrete_ok) if concrete_ok else '—'}",
                    f"බෙටන් දමන දිනයන්: {', '.join(concrete_ok) if concrete_ok else '—'}",
                ),
                tr(
                    f"High wind days — secure scaffolding/signage: {', '.join(windy_warn) if windy_warn else '—'}",
                    f"ඉහළ සුළඟ දින — ස්කැෆෝල්ඩිං/සන්ඡන ගැටළු වැළැක්වීම: {', '.join(windy_warn) if windy_warn else '—'}",
                ),
            ]

        colm = st.columns(3)
        colm[0].metric(tr("Rainy days (open)", "වැසි දවස් (විවෘත)"), rainy_days)
        colm[1].metric(tr("Clear days (open)", "පසුම්බි දවස් (විවෘත)"), clear_days)
        colm[2].metric(tr("Hot days (open)", "උණ දවස් (විවෘත)"), hot_days)

        st.write("• " + "\n• ".join(bullets) if bullets else "• " + tr("Normal operations expected.", "සාමාන්‍ය ක්‍රියාකාරීත්වය."))

        ops_text = "\n".join(
            [
                f"Business Type: {biz}",
                f"Open days: {', '.join(open_days) if open_days else '-'}",
                f"Promo days: {', '.join(best_promo) if best_promo else '-'}",
                "Operations checklist:",
                ("- Staff umbrellas/floor mats on rainy days" if rainy_days >= 1 else "- Regular entry setup"),
                ("- Promote delivery on rainy days" if rainy_days >= 1 else "- Promote in-store bundles"),
                ("- Prepare cold menu/AC checks for heat days" if hot_days >= 1 else "- Standard HVAC checks"),
            ]
        )
        st.download_button(
            "📝 " + tr("Download Weekly Ops Plan (.txt)", "වාරික ක්‍රියා සැලැස්ම බාගන්න (.txt)"),
            data=ops_text.encode("utf-8"),
            file_name="business_ops_plan_week.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Share / Export
    # ──────────────────────────────────────────────────────────────────────────
    with t_share:
        city_name = st.session_state.get("city_name", "Colombo")
        headline = tr("7-Day Weather Brief", "දින 7 ක කාලගුණ සාරාංශය")
        d1 = forecast_df.iloc[0]
        brief = (
            f"{headline} — {city_name}\n"
            f"Day1: {d1['temp']:.1f}°C, RH {d1['humidity']:.0f}%, Rain {d1['precip']:.1f}mm, "
            f"Wind {d1['windspeed']:.0f}km/h, UV {d1['uvindex']:.1f}"
        )
        flags = []
        for _, r in fx.iterrows():
            tags = []
            if r["is_rainy"]: tags.append("Rain")
            if r["is_windy"]: tags.append("Wind")
            if r["is_hot"]:   tags.append("Heat")
            if r["is_uv"]:    tags.append("UV")
            if tags: flags.append(f"{r['date']}({','.join(tags)})")
        if flags:
            brief += "\n" + tr("Flagged: ", "අවදානම්: ") + ", ".join(flags)

        st.text_area(tr("Message preview", "පණිවිඩ පෙරදසුන"), value=brief, height=140)
        wa_url = "https://wa.me/?text=" + urllib.parse.quote(brief)
        try:
            st.link_button("📲 " + tr("Share via WhatsApp", "WhatsApp හරහා බෙදාගන්න"), wa_url, use_container_width=True)
        except Exception:
            st.markdown(f"[📲 {tr('Share via WhatsApp','WhatsApp හරහා බෙදාගන්න')}]({wa_url})")

    st.success(tr("Forecast & decisions ready ✅", "අනාවැකි + තීරණ සම්පූර්ණයි ✅"), icon="🎉")

# Decide what to show now
if st.session_state.get("has_forecast") and "user_forecast_df" in st.session_state:
    render_forecast_and_decisions()
else:
    st.info(tr("Enter data and click Predict to view the 7-day forecast.",
               "දත්ත ඇතුල් කර ‘Predict’ ඔබා 7 දින අනාවැකි බලන්න."), icon="🧪")
