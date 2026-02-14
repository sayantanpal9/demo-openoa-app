import streamlit as st
import openoa
import pandas as pd

st.title("OpenOA Deployment Demo")

st.write("Version:", openoa.__version__)

data = pd.DataFrame({
    "Wind Speed": [5, 7, 9],
    "Power": [100, 200, 350]
})

st.write("Sample wind data:")
st.dataframe(data)