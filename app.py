"""Streamlit demo for US Accidents severity prediction.

Run with: streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="US Accident Severity Predictor",
                   page_icon="🚗", layout="wide")

st.title("🚗 US Accident Severity Predictor")
st.markdown(
    "Predicts whether a traffic accident will have a **severe impact on traffic flow** "
    "(Severity 3–4) based on conditions at the start of the incident.\n\n"
    "*Predicts traffic-flow impact, NOT injury severity.*"
)

MODEL_PATH = Path("models") / "lightgbm_final.joblib"
META_PATH = Path("models") / "lightgbm_final_meta.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. "
                 "Run notebooks/02_modeling.ipynb first.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH) if META_PATH.exists() else {"threshold": 0.5}
    return model, meta

model, meta = load_model()
threshold = meta.get("threshold", 0.5)

st.sidebar.header("Accident conditions")

col1, col2 = st.sidebar.columns(2)
with col1:
    lat = st.number_input("Latitude", value=34.05, format="%.4f")
    hour = st.slider("Hour", 0, 23, 8)
    day_of_week = st.selectbox(
        "Day of week", options=list(range(7)),
        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    month = st.slider("Month", 1, 12, 6)
    year = st.slider("Year", 2016, 2026, 2024)
with col2:
    lng = st.number_input("Longitude", value=-118.24, format="%.4f")
    temp = st.slider("Temperature (°F)", -20, 120, 70)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    visibility = st.slider("Visibility (mi)", 0.0, 20.0, 10.0)

st.sidebar.subheader("Weather")
pressure = st.sidebar.slider("Pressure (in)", 25.0, 32.0, 29.9)
wind_speed = st.sidebar.slider("Wind speed (mph)", 0.0, 50.0, 5.0)
weather = st.sidebar.selectbox(
    "Weather", ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Storm", "Other"])
wind_dir = st.sidebar.selectbox(
    "Wind direction",
    ["N","NE","E","SE","S","SW","W","NW","CALM","VAR","Unknown"])

st.sidebar.subheader("Location")
state = st.sidebar.selectbox(
    "State",
    ["CA","FL","TX","SC","NC","NY","PA","VA","OR","MN","GA","IL","OH","Other"])
sunrise_sunset = st.sidebar.selectbox("Sunrise/Sunset", ["Day", "Night"])
civil = st.sidebar.selectbox("Civil twilight", ["Day", "Night"])
nautical = st.sidebar.selectbox("Nautical twilight", ["Day", "Night"])
astro = st.sidebar.selectbox("Astronomical twilight", ["Day", "Night"])

st.sidebar.subheader("Road features (POI within ~50ft)")
poi_flags = {}
for flag in ["Crossing", "Junction", "Traffic_Signal", "Stop", "Station",
             "Railway", "Amenity", "Bump", "Give_Way", "No_Exit",
             "Roundabout", "Traffic_Calming"]:
    poi_flags[flag] = st.sidebar.checkbox(
        flag, value=(flag == "Traffic_Signal"))

is_weekend = int(day_of_week >= 5)
is_rush_hour = int(hour in [7, 8, 9, 16, 17, 18] and day_of_week < 5)

row = {
    "Start_Lat": lat, "Start_Lng": lng,
    "Temperature(F)": temp, "Humidity(%)": humidity,
    "Pressure(in)": pressure, "Visibility(mi)": visibility,
    "Wind_Speed(mph)": wind_speed,
    "Hour": hour, "DayOfWeek": day_of_week, "Month": month, "Year": year,
    "IsWeekend": is_weekend, "IsRushHour": is_rush_hour,
    "State": state, "Weather_Group": weather,
    "Sunrise_Sunset": sunrise_sunset, "Civil_Twilight": civil,
    "Nautical_Twilight": nautical, "Astronomical_Twilight": astro,
    "Wind_Direction": wind_dir,
}
row.update({k: int(v) for k, v in poi_flags.items()})
X = pd.DataFrame([row])

st.header("Prediction")

if st.button("Predict severity", type="primary"):
    proba = model.predict_proba(X)[0, 1]
    pred = "SEVERE" if proba >= threshold else "MILD"
    color = "#d32f2f" if pred == "SEVERE" else "#388e3c"

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"<div style='padding:1rem; border-radius:8px; "
            f"background:{color}; color:white;'>"
            f"<h2 style='margin:0; color:white;'>{pred}</h2>"
            f"<p style='margin:0;'>P(severe) = {proba:.1%}</p>"
            f"<p style='margin:0; font-size:0.8em;'>"
            f"(tuned threshold = {threshold:.2f})</p>"
            f"</div>", unsafe_allow_html=True)
    with col_b:
        st.metric("Rush hour", "Yes" if is_rush_hour else "No")
        st.metric("Weekend", "Yes" if is_weekend else "No")

    st.subheader("Input")
    st.dataframe(X.T.rename(columns={0: "value"}))
else:
    st.info("Set inputs in the sidebar and click **Predict severity**.")

st.markdown("---")
st.caption("Model: LightGBM trained on US Accidents (2016–2023, Sobhan Moosavi). "
           "Predicts traffic-flow impact, not injury severity.")
