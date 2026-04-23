import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ----------------------------
# GENERATE SAMPLE DATA (NO CSV)
# ----------------------------
@st.cache_data
def load_data():
    df = pd.DataFrame({
        "loan_amount": np.random.randint(50000, 500000, 500),
        "rate_of_interest": np.random.uniform(5, 15, 500),
        "income": np.random.randint(20000, 200000, 500),
        "status": np.random.randint(0, 2, 500)
    })
    return df

df = load_data()

# ----------------------------
# MODEL TRAINING
# ----------------------------
X = df[['loan_amount', 'rate_of_interest', 'income']]
y = df['status']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ----------------------------
# UI
# ----------------------------
st.title("💳 Loan Risk Predictor")

st.sidebar.header("Enter Loan Details")

loan_amount = st.sidebar.slider("Loan Amount", 50000, 500000, 200000)
rate_of_interest = st.sidebar.slider("Interest Rate (%)", 5.0, 15.0, 8.0)
income = st.sidebar.slider("Annual Income", 20000, 200000, 50000)

# ----------------------------
# USER INPUT
# ----------------------------
user_input = pd.DataFrame({
    "loan_amount": [loan_amount],
    "rate_of_interest": [rate_of_interest],
    "income": [income]
})

# ----------------------------
# PREDICTION
# ----------------------------
prediction = model.predict(user_input)[0]

if prediction == 1:
    st.error("❌ Bad Loan (High Risk)")
else:
    st.success("✅ Good Loan (Low Risk)")

# Probability
prob = model.predict_proba(user_input)[0][1]
st.write(f"**Probability of Default:** {prob:.2f}")

# ----------------------------
# INFO
# ----------------------------
st.markdown("---")
st.subheader("📊 About this App")
st.write("This app predicts whether a loan is risky or safe based on key financial inputs.")