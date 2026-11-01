
import pickle
import pandas as pd
import streamlit as st

with open("knn_diamond_price_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Diamond Price Prediction", page_icon="💎")
st.title("💎 Diamond Price Prediction")
st.write("Enter the diamond characteristics to estimate its price.")

with st.form("diamond_form"):
    carat = st.number_input("Carat", min_value=0.1, max_value=5.0, value=0.5, step=0.01)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
    depth = st.number_input("Depth", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
    table = st.number_input("Table", min_value=40.0, max_value=80.0, value=57.0, step=0.1)
    x = st.number_input("Length (x)", min_value=0.0, max_value=12.0, value=5.0, step=0.01)
    y = st.number_input("Width (y)", min_value=0.0, max_value=12.0, value=5.0, step=0.01)
    z = st.number_input("Depth (z)", min_value=0.0, max_value=10.0, value=3.0, step=0.01)
    submitted = st.form_submit_button("Predict price")

if submitted:
    row = pd.DataFrame([{
        "carat": carat, "cut": cut, "color": color, "clarity": clarity,
        "depth": depth, "table": table, "x": x, "y": y, "z": z
    }])
    price = model.predict(row)[0]
    st.success(f"Estimated diamond price: ${price:,.2f}")
