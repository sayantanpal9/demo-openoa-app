import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Wind Farm Dashboard",
    layout="wide"
)

st.title("Wind Farm Monitoring Dashboard")


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("wind_turbine_scada.csv")

    # Rename columns if needed (adjust based on your file)
    rename_map = {
        "Date/Time": "time",
        "LV ActivePower (kW)": "power",
        "Wind Speed (m/s)": "wind_speed",
        "Theoretical_Power_Curve (KWh)": "expected_power",
        "Wind Direction (°)": "wind_direction"
    }

    df.rename(columns=rename_map, inplace=True)

    df["time"] = pd.to_datetime(df["time"],dayfirst=True)

    # Generate additional turbine parameters
    np.random.seed(42)

    df["temperature"] = 30 + df["power"] * 0.005 + np.random.randn(len(df)) * 2
    df["rotor_speed"] = df["wind_speed"] * 0.8 + np.random.randn(len(df)) * 0.5

    # Generate fake turbine locations
    df["lat"] = 22.5 + np.random.randn(len(df)) * 0.05
    df["lon"] = 88.3 + np.random.randn(len(df)) * 0.05

    return df

df = load_data()

# -----------------------------
# CALCULATIONS
# -----------------------------
# Assume 10 min intervals
interval_hours = 10 / 60

AEP = df["power"].sum() * interval_hours / 1000  # MWh

expected_energy = df["expected_power"].sum() * interval_hours / 1000

efficiency = (AEP / expected_energy) * 100

total_loss = expected_energy - AEP

active_turbines = int(0.95 * 50)
total_turbines = 50

alerts = np.random.randint(0, 5)

# -----------------------------
# KPI CARDS
# -----------------------------
st.subheader("Plant Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Annual Energy Production",
    f"{AEP:.2f} MWh"
)

col2.metric(
    "Operational Efficiency",
    f"{efficiency:.2f}%"
)

col3.metric(
    "Active Turbines",
    f"{active_turbines}/{total_turbines}"
)

col4.metric(
    "Active Alerts",
    alerts
)

# -----------------------------
# ENERGY PRODUCTION CHART
# -----------------------------
st.subheader("Energy Production Analysis")

col1, col2 = st.columns(2)

with col1:
    fig_energy = px.line(
        df,
        x="time",
        y=["power", "expected_power"],
        title="Actual vs Expected Power Output",
        labels={
            "value": "Power (kW)",
            "time": "Time"
        }
    )
    st.plotly_chart(fig_energy, use_container_width=True)

with col2:
    loss_data = pd.DataFrame({
        "Category": ["Actual Energy", "Energy Loss"],
        "Value": [AEP, total_loss]
    })

    fig_loss = px.pie(
        loss_data,
        values="Value",
        names="Category",
        title="Energy Loss Analysis"
    )

    st.plotly_chart(fig_loss, use_container_width=True)

# -----------------------------
# TURBINE TECHNICAL PARAMETERS
# -----------------------------
st.subheader("Turbine Technical Parameters")

col1, col2 = st.columns(2)

with col1:
    fig_temp = px.line(
        df,
        x="time",
        y="temperature",
        title="Temperature vs Time"
    )
    st.plotly_chart(fig_temp, use_container_width=True)

with col2:
    fig_rotor = px.line(
        df,
        x="time",
        y="rotor_speed",
        title="Rotor Speed vs Time"
    )
    st.plotly_chart(fig_rotor, use_container_width=True)

# Power curve
fig_curve = px.scatter(
    df,
    x="wind_speed",
    y="power",
    title="Power Curve (Wind Speed vs Power Output)",
    labels={
        "wind_speed": "Wind Speed (m/s)",
        "power": "Power (kW)"
    }
)

st.plotly_chart(fig_curve, use_container_width=True)

# -----------------------------
# MAP VIEW
# -----------------------------
st.subheader("Geographic Turbine Locations")

map_center = [df["lat"].mean(), df["lon"].mean()]

m = folium.Map(location=map_center, zoom_start=8)

for i in range(50):
    color = "green"

    if np.random.rand() < 0.1:
        color = "red"

    folium.Marker(
        location=[df["lat"].iloc[i], df["lon"].iloc[i]],
        popup=f"Turbine {i+1}",
        icon=folium.Icon(color=color)
    ).add_to(m)

st_folium(m, width=1000, height=500)

# -----------------------------
# TURBINE DATA TABLE
# -----------------------------
st.subheader("Turbine Data")

display_df = df[[
    "time",
    "power",
    "expected_power",
    "wind_speed",
    "temperature",
    "rotor_speed"
]].tail(100)

st.dataframe(display_df, use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Dashboard developed for wind farm performance monitoring and client visualization.")