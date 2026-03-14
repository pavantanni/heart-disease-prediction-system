# Streamlit dashboard for heart disease prediction
# Talks to the FastAPI backend running on port 8000
# Start with: streamlit run heart_disease_streamlit_app.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import io
import json
from datetime import datetime

st.set_page_config(
    page_title="Heart Disease AI Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# color-coded risk badges and section titles
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #E85D24;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background: #fee2e2; color: #dc2626;
        padding: 1rem; border-radius: 12px;
        font-size: 1.5rem; font-weight: 700;
        text-align: center; border: 2px solid #dc2626;
    }
    .risk-moderate {
        background: #fef3c7; color: #d97706;
        padding: 1rem; border-radius: 12px;
        font-size: 1.5rem; font-weight: 700;
        text-align: center; border: 2px solid #d97706;
    }
    .risk-low {
        background: #dcfce7; color: #16a34a;
        padding: 1rem; border-radius: 12px;
        font-size: 1.5rem; font-weight: 700;
        text-align: center; border: 2px solid #16a34a;
    }
    .metric-card {
        background: white; padding: 1rem;
        border-radius: 10px; border: 1px solid #e5e7eb;
        text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .section-title {
        font-size: 1.3rem; font-weight: 600;
        color: #1f2937; margin: 1.5rem 0 0.8rem 0;
        border-left: 4px solid #E85D24; padding-left: 10px;
    }
    div[data-testid="stMetricValue"] { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)
    st.markdown("## ❤️ HeartAI Predictor")
    st.markdown("*AI-powered cardiovascular risk assessment*")
    st.markdown("### 👨‍💻 Built by Pavan Tanni")
    st.markdown("B.Tech IT | VR Siddhartha Engineering College")
    st.divider()

    page = st.radio(
    "Navigation",
    ["🩺 Risk Assessment", "📊 Batch Analysis", "🗃️ Patient History", "📈 Analytics", "ℹ️ Model Info"],
    label_visibility="collapsed",
)
    st.divider()
    st.markdown("**Model Info**")
    st.markdown("- 5-model Voting Ensemble")
    st.markdown("- ~96% AUC on UCI dataset")
    st.markdown("- SHAP Explainability")
    st.markdown("- FastAPI Backend")
    st.divider()
    st.caption("For research purposes only. Not a substitute for medical advice.")


def make_risk_gauge(probability: float) -> go.Figure:
    # green zone < 40%, yellow 40-70%, red > 70%
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={"text": "Heart Disease Risk %", "font": {"size": 18}},
        delta={"reference": 50, "increasing": {"color": "#dc2626"}, "decreasing": {"color": "#16a34a"}},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#666"},
            "bar": {"color": "#E85D24" if probability > 0.5 else "#16a34a", "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [0, 40], "color": "#dcfce7"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "#dc2626", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        }
    ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#1f2937",
    )
    return fig


def make_feature_importance_chart(risk_factors: list) -> go.Figure:
    # red bars = increases risk, green = decreases risk
    features = [rf["feature"] for rf in risk_factors]
    impacts = [rf["impact"] for rf in risk_factors]
    colors = ["#dc2626" if i > 0 else "#16a34a" for i in impacts]

    fig = go.Figure(go.Bar(
        x=impacts,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{i:+.3f}" for i in impacts],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Contributions to Prediction",
        xaxis_title="SHAP Impact",
        height=320,
        margin=dict(l=20, r=60, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#666"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def generate_pdf_report(patient_data: dict, prediction: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(232, 93, 36)
    pdf.cell(0, 12, "Heart Disease Risk Assessment Report", ln=True, align="C")

    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Patient ID: {prediction.get('patient_id', 'N/A')}", ln=True, align="C")
    pdf.ln(6)

    risk = prediction.get("risk_level", "UNKNOWN")
    risk_colors = {"HIGH": (220, 38, 38), "MODERATE": (217, 119, 6), "LOW": (22, 163, 74)}
    r, g, b = risk_colors.get(risk, (100, 100, 100))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 12, f"RISK LEVEL: {risk}   |   Probability: {prediction.get('probability', 0)*100:.1f}%", ln=True, align="C", fill=True)
    pdf.ln(8)

    pdf.set_text_color(31, 41, 55)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient Clinical Data", ln=True)
    pdf.line(pdf.get_x(), pdf.get_y(), 190, pdf.get_y())
    pdf.ln(3)

    feature_labels = {
        "age": "Age", "sex": "Sex (1=Male)", "cp": "Chest Pain Type",
        "trestbps": "Resting BP (mmHg)", "chol": "Cholesterol (mg/dl)",
        "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
        "thalach": "Max Heart Rate", "exang": "Exercise Angina",
        "oldpeak": "ST Depression", "slope": "ST Slope",
        "ca": "Major Vessels", "thal": "Thalassemia"
    }

    pdf.set_font("Arial", "", 10)
    col_width = 90
    for i, (key, label) in enumerate(feature_labels.items()):
        if i % 2 == 0 and i > 0:
            pdf.ln()
        pdf.set_font("Arial", "B", 10)
        pdf.cell(col_width // 2, 6, f"{label}:", border=0)
        pdf.set_font("Arial", "", 10)
        pdf.cell(col_width // 2, 6, str(patient_data.get(key, "N/A")), border=0)
    pdf.ln(12)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Clinical Recommendation", ln=True)
    pdf.line(pdf.get_x(), pdf.get_y(), 190, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Arial", "", 10)
    rec = prediction.get("recommendation", "")
    pdf.multi_cell(0, 6, rec)
    pdf.ln(6)

    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 5, "DISCLAIMER: This report is for research and educational purposes only. Not a substitute for professional medical advice.")

    return bytes(pdf.output(dest="S"))


if "🩺 Risk Assessment" in page:
    st.markdown('<div class="main-header">❤️ Heart Disease Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered cardiovascular risk prediction using clinical biomarkers</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-title">Patient Clinical Features</div>', unsafe_allow_html=True)

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (years)", min_value=20, max_value=100, value=55)
                sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
                cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                                  format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"][x])
                trestbps = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=220, value=130)
                chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
                fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                                       format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])

            with c2:
                thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
                exang = st.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.5, step=0.1)
                slope = st.selectbox("ST Slope", options=[0, 1, 2],
                                     format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
                ca = st.selectbox("Major Vessels (fluoroscopy)", options=[0, 1, 2, 3])
                thal = st.selectbox("Thalassemia", options=[1, 2, 3],
                                    format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x - 1])

            submitted = st.form_submit_button("Predict Heart Disease Risk", use_container_width=True, type="primary")

    with col_result:
        if submitted:
            patient_data = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
                "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
            }

            with st.spinner("Analyzing patient data..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=patient_data, timeout=10)
                    if response.status_code == 200:
                        result = response.json()

                        st.plotly_chart(make_risk_gauge(result["probability"]), use_container_width=True)

                        risk = result["risk_level"]
                        risk_class = f"risk-{risk.lower()}"
                        st.markdown(f'<div class="{risk_class}">{risk} RISK — {result["probability"]*100:.1f}% probability</div>', unsafe_allow_html=True)
                        st.markdown(f"**Recommendation:** {result['recommendation']}")

                        st.divider()

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Prediction", "Disease" if result["prediction"] else "No Disease")
                        m2.metric("Probability", f"{result['probability']*100:.1f}%")
                        m3.metric("Confidence", f"{result['confidence']*100:.1f}%")

                        st.plotly_chart(make_feature_importance_chart(result["risk_factors"]), use_container_width=True)

                        pdf_bytes = generate_pdf_report(patient_data, result)
                        st.download_button(
                            "Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"heart_report_{result['patient_id']}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

                    else:
                        st.error(f"API Error: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")

        else:
            st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
            st.markdown("""
            1. **Fill in** the 13 clinical features on the left
            2. **Click Predict** to get instant AI-powered risk assessment
            3. **Review** the risk gauge, probability, and top contributing factors
            4. **Download** a professional PDF report

            ---
            **Model:** Soft Voting Ensemble (LR + RF + XGBoost + SVM + MLP)
            **Dataset:** UCI Cleveland Heart Disease (303 patients)
            **Performance:** ~96% AUC, ~93% Accuracy
            **Explainability:** SHAP feature importance
            """)


elif "📊 Batch Analysis" in page:
    st.title("Batch Patient Analysis")
    st.markdown("Upload a CSV file with multiple patients to predict all at once.")

    # sample template so users know the expected format
    sample_df = pd.DataFrame([
        {"age": 55, "sex": 1, "cp": 1, "trestbps": 130, "chol": 250, "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2},
        {"age": 67, "sex": 0, "cp": 0, "trestbps": 160, "chol": 286, "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 1, "ca": 3, "thal": 2},
        {"age": 45, "sex": 1, "cp": 3, "trestbps": 120, "chol": 180, "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0, "oldpeak": 0.2, "slope": 0, "ca": 0, "thal": 1},
    ])
    st.download_button("Download Sample CSV Template", data=sample_df.to_csv(index=False),
                       file_name="sample_patients.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload Patient CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner(f"Analyzing {len(df)} patients..."):
                try:
                    files = {"file": ("patients.csv", uploaded.getvalue(), "text/csv")}
                    response = requests.post(f"{API_URL}/predict/batch", files=files, timeout=30)
                    if response.status_code == 200:
                        result = response.json()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Patients", result["total_patients"])
                        c2.metric("High Risk", result["high_risk"])
                        c3.metric("Moderate Risk", result["moderate_risk"])
                        c4.metric("Low Risk", result["low_risk"])

                        results_df = pd.DataFrame(result["predictions"])
                        st.dataframe(results_df, use_container_width=True)

                        fig = px.pie(
                            values=[result["high_risk"], result["moderate_risk"], result["low_risk"]],
                            names=["High Risk", "Moderate Risk", "Low Risk"],
                            color_discrete_sequence=["#dc2626", "#d97706", "#16a34a"],
                            title="Patient Risk Distribution",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.download_button("Download Results CSV", data=results_df.to_csv(index=False),
                                           file_name="batch_predictions.csv", mime="text/csv")
                    else:
                        st.error(f"API Error: {response.status_code} — {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API.")


elif "📈 Analytics" in page:
    st.title("Prediction Analytics")
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            if "message" in stats:
                st.info("No predictions made yet. Go to Risk Assessment to get started.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Predictions", stats["total_predictions"])
                c2.metric("Avg Probability", f"{stats['average_probability']*100:.1f}%")
                c3.metric("High Risk %", f"{stats['high_risk_percentage']}%")

                dist = stats["risk_distribution"]
                fig = px.bar(
                    x=list(dist.keys()), y=list(dist.values()),
                    color=list(dist.keys()),
                    color_discrete_map={"HIGH": "#dc2626", "MODERATE": "#d97706", "LOW": "#16a34a"},
                    title="Risk Level Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)
    except requests.exceptions.ConnectionError:
        st.error("API not available.")


elif "ℹ️ Model Info" in page:
    st.title("Model Information")
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            c1, c2 = st.columns(2)
            with c1:
                st.json(info)
            with c2:
                st.markdown("""
                ### Architecture
                **Soft Voting Ensemble** combines predictions from 5 models using predicted probabilities — more robust than hard majority voting.

                ### Explainability
                - **SHAP**: Shows each feature's contribution to the prediction
                - **LIME**: Local linear approximation for individual explanations
                - **Feature Importance**: Random Forest impurity-based ranking

                ### MLOps
                - MLflow experiment tracking
                - Docker containerization
                - Model versioning with joblib
                """)
    except requests.exceptions.ConnectionError:
        st.warning("API not running.")
        st.markdown("""
        **Model:** Soft Voting Ensemble
        **Components:** Logistic Regression, Random Forest, XGBoost, SVM, Neural Network
        **Dataset:** UCI Cleveland Heart Disease (303 patients, 13 features)
        **Performance:** ~96% AUC, ~93% Accuracy
        """)

elif "🗃️ Patient History" in page:
    st.title("Patient History")
    st.markdown("All predictions stored in the database.")
    try:
        response = requests.get(f"{API_URL}/patients", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data["total"] == 0:
                st.info("No predictions stored yet.")
            else:
                st.markdown(f"**Total records: {data['total']}**")

                # header row
                h1, h2, h3, h4, h5, h6 = st.columns([2, 2, 2, 2, 1, 1])
                h1.markdown("**Patient ID**")
                h2.markdown("**Risk Level**")
                h3.markdown("**Probability**")
                h4.markdown("**Timestamp**")
                h5.markdown("**PDF**")
                h6.markdown("**Action**")
                st.divider()

                for patient in data["patients"]:
                    col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 1, 1])

                    with col1:
                        st.markdown(patient['patient_id'])
                    with col2:
                        risk = patient["risk_level"]
                        color = "🔴" if risk == "HIGH" else "🟡" if risk == "MODERATE" else "🟢"
                        st.markdown(f"{color} {risk}")
                    with col3:
                        st.markdown(f"{patient['probability']*100:.1f}%")
                    with col4:
                        st.markdown(patient['timestamp'][:16].replace("T", " "))
                    with col5:
                        # rebuild the 13 features from stored DB columns
                        patient_data = {
                            "age": patient["age"], "sex": patient["sex"],
                            "cp": patient["cp"], "trestbps": patient["trestbps"],
                            "chol": patient["chol"], "fbs": patient["fbs"],
                            "restecg": patient["restecg"], "thalach": patient["thalach"],
                            "exang": patient["exang"], "oldpeak": patient["oldpeak"],
                            "slope": patient["slope"], "ca": patient["ca"],
                            "thal": patient["thal"],
                        }
                        pdf_bytes = generate_pdf_report(patient_data, patient)
                        st.download_button(
                            label="PDF",
                            data=pdf_bytes,
                            file_name=f"heart_report_{patient['patient_id']}.pdf",
                            mime="application/pdf",
                            key=f"pdf_{patient['patient_id']}",
                        )
                    with col6:
                        if st.button("Delete", key=f"del_{patient['patient_id']}", type="secondary"):
                            del_response = requests.delete(
                                f"{API_URL}/patient/{patient['patient_id']}", timeout=5
                            )
                            if del_response.status_code == 200:
                                st.success("Deleted!")
                                st.rerun()
                            else:
                                st.error("Failed to delete.")


    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")
        
st.divider()
st.caption("HeartAI Predictor | Streamlit + FastAPI + scikit-learn + XGBoost + SHAP | For educational purposes only")
