
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import tensorflow

model = tensorflow.keras.models.load_model("model.h5")

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("ohe.pkl", "rb") as file:
    ohe = pickle.load(file)

with open("lencoder.pkl", "rb") as file:
    le = pickle.load(file)


st.title("Bank Customer Churn Prediction")

st.write("Enter customer details to predict churn.")

c_score = st.number_input("Credit Score", value=650)
geo = st.selectbox("Geography", ohe.categories_[0])
sex = st.selectbox("Gender", le.classes_)
age = st.slider("Age", 18, 75, 35)
tenure = st.slider("Tenure", 0, 10, 5)
bal = st.number_input("Balance", min_value=0.0)
qty = st.slider("Num Of Products", 1, 4, 1)
cc_card = st.selectbox("Credit Card", [0, 1])
active = st.selectbox("Active Member", [0, 1])
salary = st.number_input("Estimated Salary", min_value=0.0)


ip_data = {
    "CreditScore": c_score,
    "Geography": geo,
    "Gender": sex,
    "Age": age,
    "Tenure": tenure,
    "Balance": bal,
    "NumOfProducts": qty,
    "HasCrCard": cc_card,
    "IsActiveMember": active,
    "EstimatedSalary": salary
}


if st.button("Predict Churn"):

    ip_df = pd.DataFrame([ip_data])
    ip_df["Gender"] = le.transform(ip_df["Gender"])

    geo_en_df = pd.DataFrame(
        ohe.transform(ip_df[["Geography"]]).toarray(),
        columns=ohe.get_feature_names_out(["Geography"]),
        index=ip_df.index
    )

    ip_df = pd.concat(
        [ip_df.drop(columns=["Geography"]), geo_en_df],
        axis=1
    )

    ip_df = ip_df.reindex(columns=scaler.feature_names_in_)

    ip_scaled = scaler.transform(ip_df)

    prediction = model.predict(ip_scaled, verbose=0)

    probability = float(prediction[0][0])

    st.subheader("Prediction Results")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if probability >= 0.5:
        st.error("Customer is likely to leave the bank.")
    else:
        st.success("Customer is likely to stay with the bank.")