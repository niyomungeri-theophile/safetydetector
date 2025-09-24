import streamlit as st
import pandas as pd
import joblib

# --- Custom CSS for background color ---
st.markdown(
    """
    <style>
    body {
        background-color: lightgreen;  /* Light green */
    }
    *{
    background-color:darkgreen;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load the trained model ---
MODEL_FILE = "model.joblib"  # change if your model file has a different name
model = joblib.load(MODEL_FILE)

# --- App title ---
st.title("Poultry Health Prediction Dashboard")
st.write("Enter Temperature, Humidity, and Oxygen to get the model prediction.")

# --- Input fields ---
col1, col2, col3 = st.columns(3)
temperature = col1.number_input("Temperature (°C)", value=37.0)
humidity = col2.number_input("Humidity (%)", value=60.0)
oxygen = col3.number_input("Oxygen (%)", value=21.0)

if st.button("Predict"):
    X = pd.DataFrame([[temperature, humidity, oxygen]],
                     columns=["Temperature", "humidity", "oxygen"])
    prediction = model.predict(X)[0]
    st.success(f"Model prediction: **{prediction}**")

st.markdown("---")

# --- Batch Prediction with CSV ---
st.subheader("Batch Prediction with CSV")
st.write("Upload a CSV file containing columns: Temperature, humidity, oxygen")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    # Check required columns
    required = {"Temperature", "humidity", "oxygen"}
    if not required.issubset(df.columns):
        st.error(f"CSV must have columns: {required}")
    else:
        preds = model.predict(df[["Temperature", "humidity", "oxygen"]])
        df["prediction"] = preds
        st.success("Predictions completed!")
        st.dataframe(df.head())
        st.download_button(
            "Download Predictions",
            df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
