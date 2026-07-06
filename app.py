import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClaimSense AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6ee7f7, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6ee7f7;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .flag-high {
        background: #3b1219;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .flag-medium {
        background: #2d2010;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .flag-low {
        background: #0f2818;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .agent-box {
        background: #13192b;
        border: 1px solid #2d3a5e;
        border-radius: 10px;
        padding: 1.2rem;
        font-size: 0.92rem;
        color: #e2e8f0;
        line-height: 1.7;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        border-bottom: 1px solid #2d3148;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stSidebar"] {
        background-color: #141622;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6ee7f7, #a78bfa);
        color: #0f1117;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ── Groq client ───────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# ── Synthetic data generator ──────────────────────────────────────────────────
@st.cache_data
def generate_claims(n=300, fraud_rate=0.12, seed=42):
    np.random.seed(seed)
    n_fraud = int(n * fraud_rate)
    n_normal = n - n_fraud

    procedures = ["99213", "99214", "99215", "90837", "70553", "93000", "36415", "99232"]
    diagnoses  = ["Z00.00", "M54.5", "J06.9", "E11.9", "I10", "F32.1", "K21.0", "G43.909"]
    providers  = [f"PRV{str(i).zfill(4)}" for i in range(1, 31)]
    payers     = ["BlueCross", "Aetna", "UnitedHealth", "Cigna", "Humana"]

    def make_records(n_rows, is_fraud):
        if is_fraud:
            amounts      = np.random.uniform(4500, 18000, n_rows)
            units        = np.random.randint(8, 25, n_rows)
            days_to_sub  = np.random.randint(0, 2, n_rows)
            duplicates   = np.random.randint(2, 6, n_rows)
            diff_prov    = np.random.randint(3, 8, n_rows)
        else:
            amounts      = np.random.uniform(150, 3500, n_rows)
            units        = np.random.randint(1, 6, n_rows)
            days_to_sub  = np.random.randint(3, 45, n_rows)
            duplicates   = np.random.randint(0, 2, n_rows)
            diff_prov    = np.random.randint(0, 3, n_rows)

        return pd.DataFrame({
            "claim_id":          [f"CLM{np.random.randint(100000,999999)}" for _ in range(n_rows)],
            "provider_id":       np.random.choice(providers, n_rows),
            "payer":             np.random.choice(payers, n_rows),
            "procedure_code":    np.random.choice(procedures, n_rows),
            "diagnosis_code":    np.random.choice(diagnoses, n_rows),
            "claim_amount":      np.round(amounts, 2),
            "units_billed":      units,
            "days_to_submit":    days_to_sub,
            "duplicate_claims":  duplicates,
            "diff_providers_30d":diff_prov,
            "is_fraud":          int(is_fraud),
        })

    df = pd.concat([make_records(n_normal, False), make_records(n_fraud, True)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

# ── ML pipeline ───────────────────────────────────────────────────────────────
FEATURES = ["claim_amount", "units_billed", "days_to_submit", "duplicate_claims", "diff_providers_30d"]

@st.cache_data
def run_ml(df):
    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    iso = IsolationForest(contamination=0.12, random_state=42)
    df["anomaly_flag"] = iso.fit_predict(X_scaled)          # -1 = anomaly
    df["anomaly_score"] = -iso.score_samples(X_scaled)      # higher = more anomalous

    # XGBoost risk classifier (supervised)
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    model = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.1,
                               use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    df["risk_score"] = model.predict_proba(X_scaled)[:, 1]

    # Risk tier
    def tier(r):
        if r >= 0.70: return "HIGH"
        elif r >= 0.40: return "MEDIUM"
        else: return "LOW"
    df["risk_tier"] = df["risk_score"].apply(tier)

    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    importances = dict(zip(FEATURES, model.feature_importances_))

    return df, report, importances

# ── LLM agent ─────────────────────────────────────────────────────────────────
def explain_claim(row):
    prompt = f"""You are a healthcare payment integrity specialist AI agent at a health analytics company.

Analyze this flagged healthcare claim and provide a concise clinical decision intelligence report.

CLAIM DATA:
- Claim ID: {row['claim_id']}
- Procedure Code: {row['procedure_code']}
- Diagnosis Code: {row['diagnosis_code']}
- Claim Amount: ${row['claim_amount']:,.2f}
- Units Billed: {row['units_billed']}
- Days to Submit: {row['days_to_submit']}
- Duplicate Claims (30d): {row['duplicate_claims']}
- Different Providers (30d): {row['diff_providers_30d']}
- ML Risk Score: {row['risk_score']:.2%}
- Risk Tier: {row['risk_tier']}
- Anomaly Detected: {'Yes' if row['anomaly_flag'] == -1 else 'No'}

Provide:
1. RISK SUMMARY (1-2 sentences): What makes this claim suspicious
2. KEY RED FLAGS (bullet points): Specific data points that triggered concern  
3. RECOMMENDED ACTION: One of — Auto-Approve / Flag for Review / Escalate to Investigator
4. CLINICAL REASONING: Brief explanation tying procedure/diagnosis to billing pattern

Keep it sharp, clinical, and actionable. Use plain language a medical reviewer can act on immediately."""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content

# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown('<div class="hero-title">🏥 ClaimSense AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Agentic Clinical Decision Intelligence · Anomaly Detection · Payment Integrity</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    n_claims    = st.slider("Number of Claims", 100, 500, 300, 50)
    fraud_rate  = st.slider("Fraud Rate (%)", 5, 25, 12, 1) / 100
    top_n       = st.slider("Top Flagged Claims to Review", 3, 10, 5)
    uploaded    = st.file_uploader("Upload Your Own CSV", type=["csv"],
                                   help="Needs columns: claim_amount, units_billed, days_to_submit, duplicate_claims, diff_providers_30d")
    run_btn     = st.button("▶ Run Analysis")
    st.markdown("---")
    st.markdown("**Built for Cotiviti**  \nTopic 2: Clinical Decision Making  \nAgentic AI · XGBoost · Isolation Forest  \nLLaMA-3 70B via Groq")

# ── Load data ─────────────────────────────────────────────────────────────────
if uploaded:
    df_raw = pd.read_csv(uploaded)
    # fill missing columns with defaults for compatibility
    for col in ["is_fraud", "claim_id", "provider_id", "payer", "procedure_code", "diagnosis_code"]:
        if col not in df_raw.columns:
            df_raw[col] = 0 if col == "is_fraud" else "N/A"
    st.sidebar.success(f"Loaded {len(df_raw)} claims from file.")
else:
    df_raw = generate_claims(n_claims, fraud_rate)

df, report, importances = run_ml(df_raw)

# ── KPI row ───────────────────────────────────────────────────────────────────
high   = (df["risk_tier"] == "HIGH").sum()
medium = (df["risk_tier"] == "MEDIUM").sum()
anom   = (df["anomaly_flag"] == -1).sum()
total  = len(df)
flagged_amt = df[df["risk_tier"] == "HIGH"]["claim_amount"].sum()

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, lbl in zip(
    [c1, c2, c3, c4, c5],
    [total, high, medium, anom, f"${flagged_amt:,.0f}"],
    ["Total Claims", "HIGH Risk", "MEDIUM Risk", "Anomalies Detected", "HIGH Risk Exposure"]
):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🚨 Flagged Claims", "🤖 AI Agent Review", "📈 Model Insights"])

# ── TAB 1: Dashboard ──────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        tier_counts = df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        color_map = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}
        fig_pie = px.pie(tier_counts, names="Risk Tier", values="Count",
                         color="Risk Tier", color_discrete_map=color_map,
                         hole=0.5)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", showlegend=True,
                               margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Claim Amount vs Risk Score</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(df, x="claim_amount", y="risk_score",
                                  color="risk_tier", color_discrete_map=color_map,
                                  size="units_billed", hover_data=["claim_id", "procedure_code"],
                                  labels={"claim_amount": "Claim Amount ($)", "risk_score": "Risk Score"})
        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color="#e2e8f0", margin=dict(t=20, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-header">Anomaly Score Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(df, x="anomaly_score", color="risk_tier",
                                 color_discrete_map=color_map, nbins=40,
                                 labels={"anomaly_score": "Anomaly Score"})
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#e2e8f0", margin=dict(t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">Duplicate Claims by Risk Tier</div>', unsafe_allow_html=True)
        fig_box = px.box(df, x="risk_tier", y="duplicate_claims",
                          color="risk_tier", color_discrete_map=color_map,
                          labels={"duplicate_claims": "Duplicate Claims (30d)", "risk_tier": "Risk Tier"})
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# ── TAB 2: Flagged Claims ─────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🚨 Highest Risk Claims</div>', unsafe_allow_html=True)
    flagged = df[df["risk_tier"] == "HIGH"].sort_values("risk_score", ascending=False).head(top_n)

    for _, row in flagged.iterrows():
        st.markdown(f"""
        <div class="flag-high">
            <b>🔴 {row['claim_id']}</b> &nbsp;|&nbsp; Procedure: <b>{row['procedure_code']}</b>
            &nbsp;|&nbsp; Diagnosis: <b>{row['diagnosis_code']}</b>
            &nbsp;|&nbsp; Amount: <b>${row['claim_amount']:,.2f}</b>
            &nbsp;|&nbsp; Risk Score: <b>{row['risk_score']:.1%}</b>
            &nbsp;|&nbsp; Duplicates: <b>{row['duplicate_claims']}</b>
            &nbsp;|&nbsp; Units: <b>{row['units_billed']}</b>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">⚠️ Medium Risk Claims</div>', unsafe_allow_html=True)
    medium_claims = df[df["risk_tier"] == "MEDIUM"].sort_values("risk_score", ascending=False).head(top_n)
    for _, row in medium_claims.iterrows():
        st.markdown(f"""
        <div class="flag-medium">
            <b>🟡 {row['claim_id']}</b> &nbsp;|&nbsp; Procedure: <b>{row['procedure_code']}</b>
            &nbsp;|&nbsp; Amount: <b>${row['claim_amount']:,.2f}</b>
            &nbsp;|&nbsp; Risk Score: <b>{row['risk_score']:.1%}</b>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Full Claims Table</div>', unsafe_allow_html=True)
    display_cols = ["claim_id", "procedure_code", "diagnosis_code", "claim_amount",
                    "units_billed", "duplicate_claims", "risk_score", "risk_tier", "anomaly_flag"]
    st.dataframe(df[display_cols].sort_values("risk_score", ascending=False),
                 use_container_width=True, height=350)

# ── TAB 3: AI Agent ───────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🤖 LLaMA-3 Clinical Decision Agent</div>', unsafe_allow_html=True)
    st.markdown("Select a flagged claim for the AI agent to perform a full clinical payment integrity review.")

    high_claims = df[df["risk_tier"].isin(["HIGH", "MEDIUM"])].sort_values("risk_score", ascending=False)
    claim_options = high_claims["claim_id"].tolist()

    selected_id = st.selectbox("Select Claim for AI Review", claim_options)
    selected_row = high_claims[high_claims["claim_id"] == selected_id].iloc[0]

    col_info, col_agent = st.columns([1, 2])

    with col_info:
        st.markdown("**Claim Details**")
        details = {
            "Claim ID": selected_row["claim_id"],
            "Procedure": selected_row["procedure_code"],
            "Diagnosis": selected_row["diagnosis_code"],
            "Amount": f"${selected_row['claim_amount']:,.2f}",
            "Units Billed": selected_row["units_billed"],
            "Days to Submit": selected_row["days_to_submit"],
            "Duplicates (30d)": selected_row["duplicate_claims"],
            "Diff Providers (30d)": selected_row["diff_providers_30d"],
            "Risk Score": f"{selected_row['risk_score']:.1%}",
            "Risk Tier": selected_row["risk_tier"],
            "Anomaly": "Yes" if selected_row["anomaly_flag"] == -1 else "No",
        }
        for k, v in details.items():
            st.markdown(f"**{k}:** {v}")

    with col_agent:
        if st.button("🤖 Run AI Agent Review"):
            with st.spinner("Agent analyzing claim..."):
                explanation = explain_claim(selected_row)
            st.markdown(f'<div class="agent-box">{explanation}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Batch AI Review — Top 3 HIGH Risk Claims**")
    if st.button("▶ Run Batch Agent Review"):
        batch = df[df["risk_tier"] == "HIGH"].sort_values("risk_score", ascending=False).head(3)
        for _, row in batch.iterrows():
            with st.spinner(f"Analyzing {row['claim_id']}..."):
                exp = explain_claim(row)
            tier_class = "flag-high" if row["risk_tier"] == "HIGH" else "flag-medium"
            st.markdown(f"### 🔴 {row['claim_id']} — Risk: {row['risk_score']:.1%}")
            st.markdown(f'<div class="agent-box">{exp}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

# ── TAB 4: Model Insights ─────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">XGBoost Feature Importance</div>', unsafe_allow_html=True)
    imp_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"]).sort_values("Importance")
    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="teal")
    fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    for col, label, key in zip(
        [col_r1, col_r2, col_r3, col_r4],
        ["Precision (Fraud)", "Recall (Fraud)", "F1-Score (Fraud)", "Accuracy"],
        ["1", "1", "1", "accuracy"]
    ):
        val = report[key]["precision"] if key == "1" else (
              report[key]["recall"] if key == "1" else (
              report[key]["f1-score"] if key == "1" else report.get("accuracy", 0)))
        if key == "accuracy":
            val = report.get("accuracy", 0)
        elif label == "Precision (Fraud)":
            val = report["1"]["precision"]
        elif label == "Recall (Fraud)":
            val = report["1"]["recall"]
        else:
            val = report["1"]["f1-score"]
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val:.1%}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="agent-box">
    <b>Step 1 — Isolation Forest (Unsupervised)</b><br>
    Detects statistically anomalous claims by isolating outliers in the feature space. No labels required.
    Claims with unusual combinations of amount, units, submission timing, and duplicate patterns are flagged.<br><br>

    <b>Step 2 — XGBoost Classifier (Supervised)</b><br>
    A gradient-boosted tree model trained on labeled fraud data assigns each claim a risk probability score (0–100%).
    The model learns complex non-linear patterns invisible to rule-based systems.<br><br>

    <b>Step 3 — LLaMA-3 Clinical Agent (Agentic AI)</b><br>
    Flagged claims are passed to a LLaMA-3 70B language model via Groq. The agent performs chain-of-thought
    clinical reasoning — tying procedure codes, diagnosis codes, billing patterns, and risk scores into
    a plain-language recommendation a medical reviewer can act on immediately.<br><br>

    <b>TPO Coverage:</b> Treatment (clinical code alignment) · Payment (amount/duplicate anomalies) · Operations (submission timing, provider patterns)
    </div>
    """, unsafe_allow_html=True)
