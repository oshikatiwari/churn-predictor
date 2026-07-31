# Gender -> 1 Female 0 Male
# Churn -> 1 Yes  0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the X -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["Age", "Gender", "Tenure", "MonthlyCharges"]
ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Churn Prediction App", page_icon="📉")


def train_and_save_artifacts():
    """Retrain with SMOTE + class_weight to handle churn class imbalance.

    Dataset is heavily skewed (~883 churn vs ~117 stay), so the old model
    often ignored the minority class. SMOTE balances training; class_weight
    further helps the classifier treat both outcomes fairly.
    """
    df = pd.read_csv(ROOT / "customer_churn_data.csv")
    X = df[["Age", "Gender", "Tenure", "MonthlyCharges"]].copy()
    X["Gender"] = X["Gender"].map({"Female": 1, "Male": 0})
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Oversample minority class (Stay) so the model learns both outcomes
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_s, y_train)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_res, y_res)

    joblib.dump(scaler, ROOT / "scaler.pkl")
    joblib.dump(model, ROOT / "model.pkl")
    return scaler, model


@st.cache_resource
def load_artifacts():
    scaler_path = ROOT / "scaler.pkl"
    model_path = ROOT / "model.pkl"
    try:
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        # Rebuild if old majority-class SVC or RF (from earlier extra version)
        if type(model).__name__ not in ("LogisticRegression",):
            scaler, model = train_and_save_artifacts()
    except Exception:
        scaler, model = train_and_save_artifacts()
    return scaler, model


scaler, model = load_artifacts()

st.markdown(
    """
    <style>
    :root {
        --bg: #f6f7f9;
        --surface: #ffffff;
        --text: #111827;
        --muted: #6b7280;
        --line: #e5e7eb;
        --primary: #315a9a;
        --primary-strong: #28487a;
        --radius: 12px;
        --space-sm: 0.5rem;
        --space-md: 0.9rem;
        --space-lg: 1.25rem;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stAppViewContainer"] {
        background: var(--bg);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 1.8rem !important;
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.01em;
    }

    h1 {
        margin-bottom: 0.25rem !important;
        font-weight: 700;
    }

    [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
        line-height: 1.5;
    }

    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox [data-baseweb="select"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
    }

    .stNumberInput input:focus,
    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(49, 90, 154, 0.2) !important;
    }

    .stSelectbox [data-baseweb="select"]:focus-within > div {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(49, 90, 154, 0.2) !important;
    }

    .stMetric {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: var(--space-md);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        background-color: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 10px !important;
        color: white !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1rem !important;
        min-height: 2.75rem !important;
        width: 100%;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--primary-strong) !important;
        border-color: var(--primary-strong) !important;
        color: white !important;
    }

    div.stButton > button:focus-visible {
        outline: 2px solid rgba(49, 90, 154, 0.45) !important;
        outline-offset: 2px !important;
    }

    .prediction-card {
        border: 1px solid var(--line);
        border-left: 4px solid var(--primary);
        border-radius: var(--radius);
        background: var(--surface);
        padding: var(--space-md) var(--space-lg);
        margin: var(--space-sm) 0 var(--space-lg) 0;
    }

    .prediction-helper {
        font-size: 0.9rem;
        color: var(--muted);
        margin: 0;
    }

    .prediction-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text);
        margin: 0.3rem 0 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Churn Prediction App")
st.markdown("Enter customer details to estimate churn risk.")

st.divider()

st.subheader("Customer details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=130, value=10)
with col2:
    monthly_charge = st.number_input(
        "Monthly Charge",
        min_value=30,
        max_value=150,
        value=70,
        help="Monthly service charge. Currency is unspecified in the training dataset.",
    )
    gender = st.selectbox("Gender", ["Male", "Female"])

st.caption(
    "The model considers all customer attributes. "
    "Longer tenure is generally associated with lower churn risk."
)

st.divider()

predict_button = st.button("Predict churn risk", type="primary", use_container_width=True)

if predict_button:
    gender_selected = 1 if gender == "Female" else 0
    features = pd.DataFrame(
        [[age, gender_selected, tenure, monthly_charge]],
        columns=FEATURE_COLS,
    )
    x_scaled = scaler.transform(features)
    prediction = int(model.predict(x_scaled)[0])
    probabilities = model.predict_proba(x_scaled)[0]
    stay_prob = float(probabilities[0])
    churn_prob = float(probabilities[1])

    if prediction == 1:
        outcome_icon = "⚠"
        outcome_label = "Customer is likely to churn"
    else:
        outcome_icon = "✓"
        outcome_label = "Customer is likely to stay"

    st.markdown(
        f"""
        <div class="prediction-card">
            <p class="prediction-helper">{outcome_icon} Prediction</p>
            <p class="prediction-title">{outcome_label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Prediction summary")
    m1, m2 = st.columns(2)
    m1.metric("Stay Probability", f"{stay_prob:.1%}")
    m2.metric("Churn Probability", f"{churn_prob:.1%}")

    chart_df = pd.DataFrame(
        {
            "Outcome": ["Stay", "Churn"],
            "Probability": [stay_prob, churn_prob],
        }
    )
    chart = (
        alt.Chart(chart_df)
        .mark_bar(size=60, cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "Outcome:N",
                sort=["Stay", "Churn"],
                title=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=14),
            ),
            y=alt.Y(
                "Probability:Q",
                title="Probability",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", grid=True, gridColor="#e5e7eb"),
            ),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(
                    domain=["Stay", "Churn"],
                    range=["#9ca3af", "#315a9a"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Outcome:N"),
                alt.Tooltip("Probability:Q", format=".1%"),
            ],
        )
        .properties(height=280)
        .configure_axis(labelColor="#374151", titleColor="#374151")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Enter values above, then click **Predict!**")