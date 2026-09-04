# Streamlit app for Car Price Prediction.
# Loads the already-trained pipeline (preprocessing + tuned XGBoost) — does NOT retrain.

import streamlit as st
import pandas as pd
import joblib

# Load the saved pipeline once (cached so it isn't reloaded on every interaction)
@st.cache_resource
def load_model():
    return joblib.load("models/car_price_model.pkl")

model = load_model()

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")
st.title("🚗 Car Price Prediction")
st.write(
    "Predict the resale price (in lakh INR) of a used car based on its characteristics. "
    "Model: tuned XGBoost regression pipeline."
)

# Brand list matches the mapping used during Day 1 feature engineering.
# Keeping this list in sync with the training data ensures valid one-hot categories.
brand_options = [
    "Honda", "Hyundai", "Maruti", "Toyota", "Bajaj",
    "Royal Enfield", "Hero", "Yamaha", "TVS", "KTM",
    "UM", "Hyosung", "Mahindra", "Suzuki"
]

fuel_options = ["Petrol", "Diesel"]
seller_options = ["Dealer", "Individual"]
transmission_options = ["Manual", "Automatic"]
vehicle_type_options = ["Car", "Two-Wheeler"]
# --- User inputs ---
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brand_options)
    present_price = st.number_input(
        "Present Price (lakh INR)", min_value=0.0, max_value=100.0, value=5.0, step=0.1
    )
    kms_driven = st.number_input(
        "Kms Driven", min_value=0, max_value=600000, value=30000, step=1000
    )
    car_age = st.number_input(
        "Car Age (years)", min_value=0, max_value=30, value=4, step=1
    )

with col2:
    owner = st.selectbox("Owner (number of previous owners)", [0, 1, 2, 3])
    vehicle_type = st.selectbox("Vehicle Type", vehicle_type_options)
    fuel_type = st.selectbox("Fuel Type", fuel_options)
    seller_type = st.selectbox("Seller Type", seller_options)
    transmission = st.selectbox("Transmission", transmission_options)

# --- Prediction ---
if st.button("Predict Selling Price"):

    # Build a single-row DataFrame matching the exact column names/order
    # the pipeline's preprocessor expects.
    input_df = pd.DataFrame([{
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Car_Age": car_age,
        "Owner": owner,
        "Brand": brand,
        "Vehicle_Type": vehicle_type,  
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission
    }])

    prediction = model.predict(input_df)[0]

    # Guard against negative predictions, which can occasionally occur
    # near the low end of the price range with tree-based regressors.
    prediction = max(prediction, 0)

    st.success(f"Estimated Selling Price: **₹ {prediction:.2f} lakh**")

st.caption(
    "Note: predictions are estimates based on historical Indian used-car data (2003–2018) "
    "and may not reflect current market conditions."
)