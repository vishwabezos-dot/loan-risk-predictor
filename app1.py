import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="💳",
    layout="centered"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
    }
    .stApp {
        background: linear-gradient(180deg, #fff 0%, #fff5f5 100%);
    }
    .title-box {
        padding: 18px;
        border-radius: 18px;
        background: #b30000;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="title-box">
        <h1>💳 Loan Risk Predictor</h1>
        <p>Interactive Streamlit app using Loan Amount, Interest Rate, and Income</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("Loan_Default.csv")
df.columns = df.columns.str.strip().str.lower()

required_cols = ["loan_amount", "rate_of_interest", "income", "status"]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    st.error(f"Missing columns in dataset: {missing_cols}")
    st.stop()

df = df[required_cols].copy()

# ----------------------------
# CLEAN FEATURES
# ----------------------------
for col in ["loan_amount", "rate_of_interest", "income"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------
# CLEAN TARGET
# ----------------------------
# status should be binary for this app:
# 0 = Good Loan, 1 = Bad Loan
status_raw = df["status"]

if status_raw.dtype == "object":
    s = status_raw.astype(str).str.strip().str.lower()

    mapping = {
        "good": 0, "approved": 0, "approve": 0, "yes": 0, "true": 0,
        "bad": 1, "rejected": 1, "reject": 1, "no": 1, "false": 1,
        "default": 1, "denied": 1, "declined": 1
    }

    y = s.map(mapping)

    # Fallback: try numeric conversion if the labels are "0" and "1"
    if y.isna().all():
        y = pd.to_numeric(s, errors="coerce")
else:
    y = pd.to_numeric(status_raw, errors="coerce")

df["loan_status"] = y

# Keep rows with valid target only
df = df.dropna(subset=["loan_status"]).copy()

# Convert target to integer binary if needed
df["loan_status"] = df["loan_status"].astype(int)

# If target has values other than 0/1, collapse to binary:
# 0 -> Good, anything else -> Bad
unique_target_values = sorted(df["loan_status"].dropna().unique().tolist())
if not set(unique_target_values).issubset({0, 1}):
    df["loan_status"] = (df["loan_status"] != 0).astype(int)

# ----------------------------
# CHECK CLASS COUNT
# ----------------------------
class_counts = df["loan_status"].value_counts().sort_index()

st.subheader("Dataset Class Distribution")
st.write(class_counts)

if df["loan_status"].nunique() < 2:
    st.error("After cleaning, only one class is available. The model needs both Good and Bad loan samples.")
    st.stop()

# ----------------------------
# FEATURES / TARGET
# ----------------------------
X = df[["loan_amount", "rate_of_interest", "income"]].copy()
y = df["loan_status"].copy()

# ----------------------------
# IMPUTE MISSING VALUES
# ----------------------------
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# MODELS
# ----------------------------
log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = log_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("Enter Loan Details")

loan_amnt = st.sidebar.slider(
    "Loan Amount",
    min_value=float(X_imp["loan_amount"].min()),
    max_value=float(X_imp["loan_amount"].max()),
    value=float(X_imp["loan_amount"].median()),
    step=1000.0
)

rate_of_interest = st.sidebar.slider(
    "Interest Rate",
    min_value=float(X_imp["rate_of_interest"].min()),
    max_value=float(X_imp["rate_of_interest"].max()),
    value=float(X_imp["rate_of_interest"].median()),
    step=0.1
)

income = st.sidebar.slider(
    "Annual Income",
    min_value=float(X_imp["income"].min()),
    max_value=float(X_imp["income"].max()),
    value=float(X_imp["income"].median()),
    step=1000.0
)

# ----------------------------
# PREDICTION
# ----------------------------
user_input = pd.DataFrame({
    "loan_amount": [loan_amnt],
    "rate_of_interest": [rate_of_interest],
    "income": [income]
})

user_input_imp = pd.DataFrame(
    imputer.transform(user_input),
    columns=user_input.columns
)

prediction = log_model.predict(user_input_imp)[0]
prob_bad = log_model.predict_proba(user_input_imp)[0][1]

# ----------------------------
# RESULT
# ----------------------------
st.subheader("Prediction Result")

if prediction == 0:
    st.success("✅ Good Loan")
else:
    st.error("❌ Bad Loan")

st.write(f"**Probability of Bad Loan:** {prob_bad * 100:.2f}%")
st.progress(min(max(prob_bad, 0), 1))

st.write(f"**Model Accuracy:** {acc * 100:.2f}%")

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
st.subheader("What influences the decision most?")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

top_feature = importance_df.iloc[0]["Feature"]
st.info(f"The most influential factor in this model is: **{top_feature}**")

# Optional explanation
with st.expander("How to read this output"):
    st.write(
        """
        - **Good Loan** = the model predicts lower risk
        - **Bad Loan** = the model predicts higher risk
        - **Probability of Bad Loan** tells how risky the application looks
        - The bar chart shows which variable affects the prediction most
        """
    )
