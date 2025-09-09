# pages/3_User_Dashboard.py
import os, io, math, base64, numpy as np, pandas as pd
from datetime import datetime, timedelta
import streamlit as st, plotly.express as px, requests
from tensorflow.keras.models import load_model
import joblib
import streamlit as st
# ──────────────────────────────────────────────────────────────────────────────
# Page & Auth Guard
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="User Dashboard • 7-Day Forecast", page_icon="🧭", layout="wide")

# ─────────── Logout handler (after set_page_config) ───────────
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
    # Clear auth-related session keys
    for k in list(st.session_state.keys()):
        if k.startswith("auth_") or k in ("auth_ok",):
            st.session_state.pop(k, None)
    _qp_set({})  # clear query params

    # Send back to Login
    for target in ("pages/0_🔐_Login.py", "pages/0_Login.py", "🔐 Login", "login.py"):
        try:
            st.switch_page(target); st.stop()
        except Exception:
            pass
    st.stop()
# ───────────────────────────────────────────────────────────────

if not st.session_state.get("auth_ok", False):
    # redirect to Login if user isn't logged in
    for target in ("pages/0_🔐_Login.py", "pages/0_Login.py", "🔐 Login"):
        try:
            st.switch_page(target)
            st.stop()
        except Exception:
            pass
    st.warning("Please log in first.", icon="🔑")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Look & Feel
# ──────────────────────────────────────────────────────────────────────────────
def _b64_image(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

_bg64 = _b64_image("cloud2.jpg") or _b64_image(os.path.join("lstm","cloud2.jpg"))
if _bg64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{_bg64}") no-repeat center center fixed;
            background-size: cover;
        }}
        header[data-testid="stHeader"] {{
            background:#0b3d66 !important; border-bottom:1px solid rgba(255,255,255,.12);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.title("🧭 User Dashboard")
st.caption("Signed-in experience for fast 7-day predictions + decision support for agriculture and local businesses.")

# Fixed top-right actions
st.markdown("""
<style>
.top-actions{position:fixed;top:70px;right:18px;z-index:9999;display:flex;gap:8px}
.top-actions a{padding:8px 12px;border-radius:8px;border:1px solid rgba(255,255,255,.25);
text-decoration:none;color:#fff !important;box-shadow:0 2px 6px rgba(0,0,0,.12);font-weight:700}
.top-actions a.logout{background:#dc2626} .top-actions a.logout:hover{background:#b91c1c}
</style>
<div class="top-actions">
  <a class="logout" href="?logout=1">Logout</a>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Core model + features (same order as admin page)
# ──────────────────────────────────────────────────────────────────────────────
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
    df["datetime"] = pd.to_datetime(df["datetime"])
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
# Derived metrics for richer guidance
# ──────────────────────────────────────────────────────────────────────────────
def heat_index_c(t_c, rh):
    # Rothfusz approximation (converted to °C crudely)
    t_f = t_c*9/5 + 32
    hi_f = -42.379 + 2.04901523*t_f + 10.14333127*rh - .22475541*t_f*rh \
           - .00683783*t_f*t_f - .05481717*rh*rh + .00122874*t_f*t_f*rh \
           + .00085282*t_f*rh*rh - .00000199*t_f*t_f*rh*rh
    return (hi_f-32)*5/9

def wet_bulb_stull(t_c, rh):
    # Stull (2011) quick approximation
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

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — model + input mode
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Forecast Controls")
    try:
        model, scaler = load_model_and_scaler()
        st.success("Model ready", icon="✅")
    except Exception as e:
        st.error(f"Model/scaler load failed: {e}")
        st.stop()

    mode = st.radio("Input Mode", ["Enter today's values", "Upload last 30 days (CSV)"], index=0)

    st.markdown("### 🌍 (Optional) Live Weather")
    api_key = st.text_input("OpenWeatherMap API key", type="password")
    city = st.text_input("City (e.g., Colombo)")
    fetch_btn = st.button("📡 Use current city weather")

# Defaults (user-friendly)
default_values = dict(temp=28.0, humidity=75.0, precip=2.0, windspeed=12.0, winddir=180.0,
                      cloudcover=50.0, dew=22.0, uvindex=7.0, sealevelpressure=1010.0)
for k, v in default_values.items():
    st.session_state.setdefault(k, v)

# Live fetch
if fetch_btn:
    if not api_key or not city:
        st.sidebar.error("Enter API key and City.")
    else:
        try:
            r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                             params={"q": city, "appid": api_key, "units": "metric"}, timeout=10)
            r.raise_for_status()
            d = r.json()
            main, wind, clouds, rain, snow = d.get("main", {}), d.get("wind", {}), d.get("clouds", {}), d.get("rain", {}), d.get("snow", {})
            temp = float(main.get("temp", st.session_state["temp"]))
            humidity = float(main.get("humidity", st.session_state["humidity"]))
            pressure = float(main.get("pressure", st.session_state["sealevelpressure"]))
            wind_kmh = float(wind.get("speed", 0.0)) * 3.6
            wind_dir = float(wind.get("deg", st.session_state["winddir"]))
            cloud_pct = float(clouds.get("all", st.session_state["cloudcover"]))
            precip = float(rain.get("1h", 0.0) or snow.get("1h", 0.0) or 0.0)
            # UV quick attempt via OneCall (optional)
            uv = st.session_state["uvindex"]
            coord = d.get("coord", {})
            if coord:
                try:
                    r2 = requests.get("https://api.openweathermap.org/data/3.0/onecall",
                                      params={"lat": coord.get("lat"), "lon": coord.get("lon"),
                                              "appid": api_key, "units": "metric", "exclude": "minutely,hourly,daily,alerts"},
                                      timeout=10)
                    if r2.ok:
                        uv = float(r2.json().get("current", {}).get("uvi", uv))
                except Exception:
                    pass
            # Derive dewpoint from temp/rh for convenience
            a, b = 17.62, 243.12
            gamma = (a*temp)/(b+temp) + math.log(max(min(humidity,100.0),1e-2)/100.0)
            dew = (b*gamma)/(a-gamma)

            st.session_state.update(dict(
                temp=round(temp,1), humidity=round(humidity,0), precip=round(precip,1),
                windspeed=round(wind_kmh,1), winddir=round(wind_dir,0), cloudcover=round(cloud_pct,0),
                dew=round(dew,1), uvindex=round(uv,1), sealevelpressure=round(pressure,0),
                city_name=d.get("name", city)
            ))
            st.sidebar.success("Live weather applied ✅")
            st.rerun()
        except Exception as ex:
            st.sidebar.error(f"Fetch failed: {ex}")

# ──────────────────────────────────────────────────────────────────────────────
# Inputs & Forecast
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.subheader("📥 Provide Input")
    st.info("Either enter today's values or upload a CSV with the last 30 days.", icon="ℹ️")

    history_df = None
    if mode == "Upload last 30 days (CSV)":
        st.write("CSV needs `datetime` + " + ", ".join(f"`{c}`" for c in FEATURE_COLUMNS))
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            try:
                raw = up.read()
                history_df = pd.read_csv(io.BytesIO(raw))
                if "datetime" not in history_df.columns:
                    st.error("CSV must contain `datetime` column.")
                    history_df = None
                else:
                    history_df = add_date_features(history_df)
                    missing = [c for c in FEATURE_COLUMNS if c not in history_df.columns]
                    if missing:
                        st.error(f"Missing: {missing}")
                        history_df = None
                    else:
                        st.success("CSV loaded.", icon="📄")
            except Exception as e:
                st.error(f"Parse error: {e}")
                history_df = None
    else:
        c1, c2, c3 = st.columns(3)
        st.session_state["temp"] = c1.number_input("Temp (°C)", value=float(st.session_state["temp"]))
        st.session_state["humidity"] = c2.number_input("Humidity (%)", value=float(st.session_state["humidity"]), min_value=0.0, max_value=100.0)
        st.session_state["precip"] = c3.number_input("Precip (mm)", value=float(st.session_state["precip"]), min_value=0.0)

        c4, c5, c6 = st.columns(3)
        st.session_state["windspeed"] = c4.number_input("Wind Speed (km/h)", value=float(st.session_state["windspeed"]), min_value=0.0)
        st.session_state["winddir"] = c5.number_input("Wind Dir (°)", value=float(st.session_state["winddir"]), min_value=0.0, max_value=360.0)
        st.session_state["cloudcover"] = c6.number_input("Cloud Cover (%)", value=float(st.session_state["cloudcover"]), min_value=0.0, max_value=100.0)

        c7, c8, c9 = st.columns(3)
        st.session_state["dew"] = c7.number_input("Dew Point (°C)", value=float(st.session_state["dew"]))
        st.session_state["uvindex"] = c8.number_input("UV Index", value=float(st.session_state["uvindex"]), min_value=0.0)
        st.session_state["sealevelpressure"] = c9.number_input("Sea Level Pressure (hPa)", value=float(st.session_state["sealevelpressure"]))

with right:
    st.subheader("🔮 Forecast")
    go = st.button("🚀 Predict Next 7 Days", type="primary", use_container_width=True)

    if go:
        try:
            if mode == "Upload last 30 days (CSV)" and history_df is not None:
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

            # Derived columns
            forecast_df["heat_index"] = [heat_index_c(t, rh) for t, rh in zip(forecast_df["temp"], forecast_df["humidity"])]
            forecast_df["wet_bulb"]   = [wet_bulb_stull(t, rh) for t, rh in zip(forecast_df["temp"], forecast_df["humidity"])]
            forecast_df["rain_class"] = forecast_df["precip"].apply(rain_class)
            forecast_df["uv_risk"]    = forecast_df["uvindex"].apply(uv_risk)
            forecast_df["wind_risk"]  = forecast_df["windspeed"].apply(wind_risk)

            st.session_state["user_forecast_df"] = forecast_df.copy()

            # Quick KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Day 1 Temp (°C)", f"{forecast_df.loc[0,'temp']:.1f}")
            k2.metric("Day 1 RH (%)", f"{forecast_df.loc[0,'humidity']:.0f}")
            k3.metric("Day 1 Rain (mm)", f"{forecast_df.loc[0,'precip']:.1f}")
            k4.metric("Day 1 Wind (km/h)", f"{forecast_df.loc[0,'windspeed']:.0f}")

            # Tabs: Overview | Agriculture | Business
            t_over, t_agri, t_biz = st.tabs(["📈 Overview", "🌾 Agriculture", "🏪 Local Business"])

            with t_over:
                template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
                plot_df = forecast_df.melt(id_vars="date",
                                           value_vars=["temp","humidity","precip","windspeed","uvindex"],
                                           var_name="feature", value_name="value")
                fig = px.line(plot_df, x="date", y="value", color="feature", markers=True,
                              title="Key Weather Features (7 days)", template=template)
                fig.update_layout(hovermode="x unified", legend_title_text="Feature")
                st.plotly_chart(fig, use_container_width=True, theme=None)

                st.dataframe(forecast_df.round(2), use_container_width=True, hide_index=True)

                # Per-day summaries with richer cues
                st.markdown("**🧾 Day-by-day summary:**")
                for _, r in forecast_df.iterrows():
                    desc = f"{r['date']}: {r['temp']:.1f}°C ({r['heat_index']:.1f}°C HI), RH {r['humidity']:.0f}%, rain {r['precip']:.1f} mm ({r['rain_class']}), wind {r['windspeed']:.0f} km/h ({r['wind_risk']}), UV {r['uvindex']:.1f} ({r['uv_risk']})"
                    st.write("• " + desc)

                st.download_button("⬇️ Download Forecast (CSV)",
                                   data=forecast_df.to_csv(index=False).encode("utf-8"),
                                   file_name="user_forecast_7day.csv",
                                   mime="text/csv", use_container_width=True)

            with t_agri:
                st.markdown("**Field Guidance (auto)**")
                total_rain = float(forecast_df["precip"].clip(lower=0).sum())
                tavg = float(forecast_df["temp"].mean())
                wind_max = float(forecast_df["windspeed"].max())
                uv_max = float(forecast_df["uvindex"].max())

                # Simple weekly assessment
                notes = []
                if total_rain < 30:   notes += ["Low rain → plan irrigation 2–3 sessions / week"]
                elif total_rain < 70: notes += ["Moderate rain → 1 session may be enough"]
                else:                  notes += ["High rain → watch waterlogging; improve drainage"]

                if tavg >= 30: notes += ["Heat stress likely — sow/transplant early morning; consider 30–40% shade net for nurseries"]
                if wind_max >= 35: notes += ["Stake vines/okra; secure covers — gusty day expected"]
                if uv_max >= 8: notes += ["Midday field work: hats, sleeves, sunscreen (UV ‘very high’+)"]

                # Best two sowing days (low rain + comfy temp)
                tmp = forecast_df.copy()
                tmp["score"] = (tmp["precip"].max()-tmp["precip"])/max(tmp["precip"].max(),1e-6) \
                               + (1 - (abs(tmp["temp"]-28)/10).clip(0,1))
                best_days = tmp.nlargest(2, "score")["date"].tolist()

                st.write("• " + "\n• ".join(notes))
                st.write(f"**Best sowing/transplant windows:** {', '.join(best_days) if best_days else '—'}")

                # Quick risk badges
                colA, colB, colC, colD = st.columns(4)
                colA.metric("Rain (7d)", f"{total_rain:.0f} mm")
                colB.metric("Avg Temp", f"{tavg:.1f} °C")
                colC.metric("Max Wind", f"{wind_max:.0f} km/h")
                colD.metric("Max UV", f"{uv_max:.1f}")

            with t_biz:
                st.markdown("**Operational Guidance (auto)**")
                df = forecast_df.copy()
                df["is_rainy"] = df["precip"] >= 10
                df["is_clear"] = (df["cloudcover"] <= 30) & (~df["is_rainy"])
                df["is_hot"] = df["heat_index"] >= 32
                rainy_days = int(df["is_rainy"].sum())
                clear_days = int(df["is_clear"].sum())
                hot_days = int(df["is_hot"].sum())

                # Best two promo days
                tmp = df.copy()
                tmp["rain_score"] = (tmp["precip"].max() - tmp["precip"]) / max(tmp["precip"].max(), 1e-6)
                tmp["temp_score"] = 1 - (abs(tmp["temp"]-28)/10).clip(0,1)
                tmp["footfall_score"] = 0.65*tmp["rain_score"] + 0.35*tmp["temp_score"]
                best_promo = tmp.nlargest(2, "footfall_score")["date"].tolist()

                bullets = []
                if rainy_days >= 2:
                    bullets += ["Staff umbrellas/floor mats at entrance; promote delivery offer on rain days"]
                if clear_days >= 2:
                    bullets += [f"In-store events/promos on: {', '.join(best_promo)}"]
                if hot_days >= 2:
                    bullets += ["Boost cold drinks/desserts; keep indoor comfort high"]

                st.write("• " + "\n• ".join(bullets) if bullets else "• Normal operations expected.")

                b1,b2,b3 = st.columns(3)
                b1.metric("Rainy days (≥10mm)", rainy_days)
                b2.metric("Clear days", clear_days)
                b3.metric("Hot discomfort days", hot_days)

            st.success("Forecast ready ✅", icon="🎉")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
