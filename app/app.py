import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# =========================================================
# CONFIG
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODELS = ROOT / "models"

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes",
    "Turbidity"
]

RANDOM_STATE = 42

BOUNDS = {
    "ph": (0, 14, 7, "pH"),
    "Hardness": (0, 400, 196, "Hardness · mg/L"),
    "Solids": (0, 60000, 20000, "Solids · ppm"),
    "Chloramines": (0, 15, 7, "Chloramines · ppm"),
    "Sulfate": (0, 500, 330, "Sulfate · mg/L"),
    "Conductivity": (0, 800, 420, "Conductivity · µS/cm"),
    "Organic_carbon": (0, 30, 14, "Organic Carbon · ppm"),
    "Trihalomethanes": (0, 130, 66, "Trihalomethanes · µg/L"),
    "Turbidity": (0, 7, 4, "Turbidity · NTU"),
}

st.set_page_config(
    page_title="AquaAI — Water Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ===== LIGHT TEXT FOR DARK UI ===== */

/* AI Prediction input labels */
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] label p {
    color: #dff7ff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Input boxes */
[data-testid="stNumberInput"] input {
    color: #f4fbff !important;
    background-color: #0d1d2b !important;
    border-color: rgba(100, 210, 255, 0.25) !important;
}

/* Dataset Overview metric labels */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #b9d4e2 !important;
    font-weight: 600 !important;
}

/* Dataset Overview metric values */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #f4fbff !important;
    font-weight: 700 !important;
}

/* Metric delta/helper text */
[data-testid="stMetricDelta"] {
    color: #9fc2d2 !important;
}

/* Selectbox labels */
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label p {
    color: #dff7ff !important;
    font-weight: 600 !important;
}

/* Selectbox selected value */
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    color: #f4fbff !important;
}

/* General Streamlit labels */
label {
    color: #dff7ff !important;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0,220,255,.09), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(0,120,255,.08), transparent 30%),
        linear-gradient(135deg, #050a12 0%, #07111d 50%, #03070d 100%);
    color: #eaf7ff;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent !important;}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f4fbff !important;
}

.hero {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(85,220,255,.18);
    border-radius: 28px;
    padding: 52px 55px;
    margin-bottom: 24px;
    background:
    linear-gradient(135deg, rgba(11,28,43,.95), rgba(5,14,25,.88));
    box-shadow: 0 25px 80px rgba(0,0,0,.35);
}

.hero:before {
    content: "";
    position: absolute;
    width: 350px;
    height: 350px;
    right: -100px;
    top: -140px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,220,255,.28), transparent 68%);
    animation: pulse 5s infinite ease-in-out;
}

.hero:after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    right: 80px;
    bottom: -140px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,130,255,.20), transparent 70%);
}

@keyframes pulse {
    0%,100% {transform: scale(1); opacity:.7;}
    50% {transform: scale(1.2); opacity:1;}
}

.hero-content {
    position: relative;
    z-index: 2;
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 50px;
    border: 1px solid rgba(80,220,255,.3);
    background: rgba(0,210,255,.07);
    color: #65e5ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(42px, 6vw, 76px);
    line-height: .98;
    font-weight: 800;
    margin: 0;
    letter-spacing: -3px;
    background: linear-gradient(90deg, #ffffff, #67e8f9, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    max-width: 700px;
    margin-top: 20px;
    color: #91a8b8;
    font-size: 17px;
    line-height: 1.7;
}

.status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 25px;
    color: #9be7c4;
    font-size: 13px;
    font-weight: 600;
}

.dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #38e6a2;
    box-shadow: 0 0 15px #38e6a2;
    animation: blink 1.8s infinite;
}

@keyframes blink {
    0%,100% {opacity:1;}
    50% {opacity:.35;}
}

.metric-card {
    background: linear-gradient(145deg, rgba(17,34,50,.88), rgba(7,17,28,.9));
    border: 1px solid rgba(130,210,255,.13);
    border-radius: 18px;
    padding: 23px;
    min-height: 125px;
    box-shadow: 0 15px 40px rgba(0,0,0,.18);
    transition: .25s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(80,220,255,.35);
}

.metric-label {
    color: #718899;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.metric-value {
    color: #f4fbff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 31px;
    font-weight: 700;
    margin-top: 8px;
}

.metric-accent {
    color: #54ddff;
}

.section {
    margin-top: 38px;
    margin-bottom: 18px;
}

.glass {
    background: rgba(11,25,38,.72);
    border: 1px solid rgba(130,210,255,.12);
    border-radius: 20px;
    padding: 25px;
}

div[data-testid="stTabs"] button {
    color: #7e98a9 !important;
    font-weight: 600;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #62e4ff !important;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(72,220,255,.35);
    background: linear-gradient(90deg, #087d9e, #1467b7);
    color: white;
    font-weight: 700;
    padding: 12px;
    transition: .2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,180,255,.2);
}

[data-testid="stMetric"] {
    background: rgba(12,28,42,.8);
    border: 1px solid rgba(100,200,255,.12);
    padding: 18px;
    border-radius: 16px;
}

.stAlert {
    border-radius: 14px;
}

.footer {
    text-align: center;
    color: #526979;
    margin-top: 60px;
    padding-top: 25px;
    border-top: 1px solid rgba(120,180,210,.1);
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    model = joblib.load(MODELS / "best_model.pkl")
    scaler = joblib.load(MODELS / "scaler.pkl")
    imputer = joblib.load(MODELS / "imputer.pkl")

    with open(MODELS / "feature_order.json") as f:
        feature_order = json.load(f)

    return model, scaler, imputer, feature_order


@st.cache_data
def evaluate(df):
    X = df[FEATURES]
    y = df["Potability"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    model, scaler, imputer, features = load_model()

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=features
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=features
    )

    pred = model.predict(X_test)

    proba = None

    if hasattr(model, "predict_proba"):
        idx = list(model.classes_).index(1)
        proba = model.predict_proba(X_test)[:, idx]

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba) if proba is not None else None
    }

    return y_test, pred, proba, metrics


def model_name(model):
    names = {
        "DecisionTreeClassifier": "Decision Tree",
        "LogisticRegression": "Logistic Regression",
        "KNeighborsClassifier": "K-Nearest Neighbors"
    }
    return names.get(model.__class__.__name__, model.__class__.__name__)


# =========================================================
# HERO
# =========================================================

df = load_data()
model, scaler, imputer, feature_order = load_model()
y_test, y_pred, y_proba, metrics = evaluate(df)

st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="badge">AI · WATER INTELLIGENCE · CAPSTONE</div>
        <div class="hero-title">AquaAI</div>
        <div style="font-size:26px;font-weight:600;margin-top:8px;color:#d8f7ff;">
            Water Quality Intelligence Platform
        </div>
        <div class="hero-sub">
            Machine-learning powered analysis of physicochemical water
            properties for intelligent potability classification.
            Explore the data, understand model performance, and analyze
            individual water samples.
        </div>
        <div class="status">
            <span class="dot"></span>
            MODEL SYSTEM ONLINE · """ + model_name(model).upper() + """
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# TOP METRICS
# =========================================================

cols = st.columns(5)

for col, name in zip(cols, ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]):
    value = metrics[name]
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{name}</div>
            <div class="metric-value metric-accent">{value*100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)

tabs = st.tabs([
    "◈ OVERVIEW",
    "◈ ANALYTICS",
    "◈ MODEL PERFORMANCE",
    "◈ AI PREDICTION"
])


# =========================================================
# OVERVIEW
# =========================================================

with tabs[0]:

    st.markdown('<div class="section"><h2>Dataset Overview</h2></div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Samples", f"{len(df):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Missing Values", f"{df.isna().sum().sum():,}")
    c4.metric("Duplicates", f"{df.duplicated().sum():,}")

    left, right = st.columns(2)

    with left:
        st.markdown("### Potability Distribution")

        counts = df["Potability"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            x=["Not Potable", "Potable"],
            y=[counts.get(0, 0), counts.get(1, 0)],
            ax=ax
        )
        ax.set_ylabel("Samples")
        ax.set_xlabel("")
        ax.set_title("Target Class Distribution")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("### Missing Data")

        missing = df[FEATURES].isna().sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            x=missing.values,
            y=missing.index,
            ax=ax
        )
        ax.set_xlabel("Missing Values")
        ax.set_ylabel("")
        ax.set_title("Missing Values by Feature")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# =========================================================
# ANALYTICS
# =========================================================

with tabs[1]:

    st.markdown('<div class="section"><h2>Exploratory Data Analysis</h2></div>',
                unsafe_allow_html=True)

    selected = st.selectbox("Analyze feature", FEATURES)

    a, b = st.columns(2)

    with a:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            data=df,
            x=selected,
            hue="Potability",
            kde=True,
            common_norm=False,
            ax=ax
        )
        ax.set_title(f"{selected} Distribution")
        st.pyplot(fig)
        plt.close(fig)

    with b:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x="Potability",
            y=selected,
            ax=ax
        )
        ax.set_title(f"{selected} vs Potability")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Feature Correlation Matrix")

    corr = df[FEATURES + ["Potability"]].corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Feature Correlation with Potability")

    target_corr = corr["Potability"].drop("Potability").sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    target_corr.plot(kind="barh", ax=ax)
    ax.set_xlabel("Correlation")
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

with tabs[2]:

    st.markdown('<div class="section"><h2>Model Performance</h2></div>',
                unsafe_allow_html=True)

    st.write(
        "Evaluation uses the same 80/20 stratified test split used "
        "during model development (random_state = 42)."
    )

    metric_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Score": list(metrics.values())
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=metric_df,
        x="Metric",
        y="Score",
        ax=ax
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name(model)} — Test Performance")

    for i, v in enumerate(metric_df["Score"]):
        ax.text(i, v + .025, f"{v:.3f}", ha="center", fontweight="bold")

    st.pyplot(fig)
    plt.close(fig)

    left, right = st.columns(2)

    with left:

        st.markdown("### Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Not Potable", "Potable"],
            yticklabels=["Not Potable", "Potable"],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
        plt.close(fig)

    with right:

        st.markdown("### Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            target_names=["Not Potable", "Potable"],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).T.round(3),
            use_container_width=True
        )

    if y_proba is not None:

        left, right = st.columns(2)

        with left:

            st.markdown("### ROC Curve")

            fpr, tpr, _ = roc_curve(y_test, y_proba)

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"AUC = {metrics['ROC-AUC']:.3f}"
            )

            ax.plot([0, 1], [0, 1], "--", label="Random")

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

        with right:

            st.markdown("### Precision–Recall Curve")

            precision, recall, _ = precision_recall_curve(
                y_test,
                y_proba
            )

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.plot(
                recall,
                precision,
                linewidth=2
            )

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")

            st.pyplot(fig)
            plt.close(fig)

    st.markdown("### Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({
            "Feature": feature_order,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.barplot(
            data=importance,
            x="Importance",
            y="Feature",
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            importance.round(4),
            use_container_width=True,
            hide_index=True
        )

    elif hasattr(model, "coef_"):

        coef = pd.DataFrame({
            "Feature": feature_order,
            "Coefficient": model.coef_[0]
        })

        coef["Importance"] = coef["Coefficient"].abs()
        coef = coef.sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.barplot(
            data=coef,
            x="Importance",
            y="Feature",
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)


# =========================================================
# PREDICTION
# =========================================================

with tabs[3]:

    st.markdown('<div class="section"><h2>AI Water Sample Analyzer</h2></div>',
                unsafe_allow_html=True)

    st.write(
        "Enter physicochemical measurements and let the trained model "
        "classify the sample."
    )

    col1, col2, col3 = st.columns(3)

    inputs = {}

    for i, feature in enumerate(feature_order):

        lo, hi, default, label = BOUNDS[feature]

        target = [col1, col2, col3][i % 3]

        inputs[feature] = target.number_input(
            label,
            min_value=float(lo),
            max_value=float(hi),
            value=float(default),
            step=0.1,
            key=f"input_{feature}"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("💧 ANALYZE WATER SAMPLE", type="primary"):

        row = pd.DataFrame(
            [[inputs[f] for f in feature_order]],
            columns=feature_order
        )

        row_imp = pd.DataFrame(
            imputer.transform(row),
            columns=feature_order
        )

        row_scaled = pd.DataFrame(
            scaler.transform(row_imp),
            columns=feature_order
        )

        prediction = model.predict(row_scaled)[0]

        probability = None

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(row_scaled)[0]
            class_index = list(model.classes_).index(1)
            probability = probs[class_index]

        st.divider()

        if prediction == 1:

            st.success("## ✓ POTABLE CLASSIFICATION")

            st.markdown(
                "The trained model classified this sample as **Potable (Class 1)**."
            )

        else:

            st.error("## ⚠ NOT POTABLE CLASSIFICATION")

            st.markdown(
                "The trained model classified this sample as **Not Potable (Class 0)**."
            )

        if probability is not None:

            c1, c2 = st.columns(2)

            c1.metric(
                "Potable Probability",
                f"{probability * 100:.2f}%"
            )

            c2.metric(
                "Not Potable Probability",
                f"{(1 - probability) * 100:.2f}%"
            )

            st.progress(float(np.clip(probability, 0, 1)))

            st.caption(
                "Probability is a machine-learning estimate and is not "
                "equivalent to laboratory or regulatory certification."
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    <b>AquaAI</b> · Water Quality Intelligence Platform<br>
    Machine Learning Capstone · Educational Prototype<br><br>
    Built with Python · Scikit-learn · Pandas · Streamlit
</div>
""", unsafe_allow_html=True)