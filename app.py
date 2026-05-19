import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("ice_cream_model.pkl")
df = pd.read_csv("data/ice_cream_sales.csv")

st.title("Ice Cream Sales Predictor")

st.write("This app predicts daily ice cream sales based on temperature.")

temperature = st.number_input(
    "Enter temperature in Celsius:",
    min_value=0.0,
    max_value=50.0,
    value=30.0
)

prediction = model.predict([[temperature]])[0]

st.metric("Predicted Ice Cream Sales", round(prediction, 2))

st.subheader("Dataset")
st.dataframe(df)

st.subheader("Temperature vs Sales")

fig, ax = plt.subplots()
ax.scatter(df["Temperature"], df["Sales"])
ax.set_xlabel("Temperature")
ax.set_ylabel("Sales")
ax.set_title("Temperature vs Ice Cream Sales")

st.pyplot(fig)