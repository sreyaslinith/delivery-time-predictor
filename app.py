import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model and feature columns
model = joblib.load("best_delivery_time_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Streamlit app
st.set_page_config(page_title="📦 Delivery Duration Predictor", layout="centered")
st.title("🚚 Predict Delivery Duration")
st.markdown("Enter order details to estimate delivery time in **seconds**.")

# Inputs
user_created_at = st.text_input("🕒 Order Created At (YYYY-MM-DD HH:MM:SS)", "2025-07-01 14:30:00")
user_order_protocol = st.selectbox("📋 Order Protocol", ["Standard platform order", "Phone order", "Partnered restaurant app"])
user_store_category = st.selectbox("🏪 Store Primary Category", ["Thai", "Pizza", "Burgers", "Fast Food", "Chineese", "Indian", "Other"])
total_items = st.number_input("🧮 Total Items", min_value=1, step=1)
subtotal = st.number_input("💰 Subtotal (₹)", min_value=0.0, step=1.0)
num_distinct_items = st.number_input("🔢 Distinct Items", min_value=1, step=1)
min_item_price = st.number_input("🧾 Minimum Item Price (₹)", min_value=0.0, step=1.0)
max_item_price = st.number_input("💵 Maximum Item Price (₹)", min_value=0.0, step=1.0)
estimated_store_to_consumer_driving_duration = st.number_input("🚗 Estimated Driving Duration (seconds)", min_value=0, step=1)

# Feature engineering
try:
    created_at = pd.to_datetime(user_created_at)
    order_hour = created_at.hour
    day_of_week = created_at.day_name()

    # Combine all features into a DataFrame
    raw_input = pd.DataFrame([{
        "order_protocol": user_order_protocol,
        "store_primary_category": user_store_category,
        "order_hour": order_hour,
        "day_of_week": day_of_week,
        "total_items": total_items,
        "subtotal": subtotal,
        "num_distinct_items": num_distinct_items,
        "min_item_price": min_item_price,
        "max_item_price": max_item_price,
        "estimated_store_to_consumer_driving_duration": estimated_store_to_consumer_driving_duration
    }])

    # One-hot encode categorical features
    encoded_input = pd.get_dummies(raw_input)

    # Align with training columns
    encoded_input = encoded_input.reindex(columns=feature_columns, fill_value=0)

    # Predict
    if st.button("🔮 Predict Delivery Duration"):
        prediction = model.predict(encoded_input)[0]
        st.success(f"📦 Estimated Delivery Duration: **{prediction:.2f} seconds**")

except Exception as e:
    st.error(f"⚠️ Invalid input: {e}")
