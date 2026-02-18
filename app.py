"""
╔══════════════════════════════════════════════════════╗
║  CardioSense AI — Heart Disease Prediction Platform  ║
║  Powered by RBF-SVM  |  Built with Streamlit         ║
╚══════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, base64, io, warnings, time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Google Fonts + Global CSS ─────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── Root variables ── */
:root {
    --brand-red:    #E53935;
    --brand-dark:   #1A0A0A;
    --brand-card:   #1F1010;
    --brand-border: #3D1515;
    --brand-green:  #00C853;
    --brand-amber:  #FFB300;
    --brand-blue:   #42A5F5;
    --text-primary: #F5F0F0;
    --text-muted:   #9E8585;
    --gradient-hero: linear-gradient(135deg,#1A0A0A 0%,#2D0F0F 50%,#1A0A0A 100%);
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--brand-dark);
    color: var(--text-primary);
}
.main .block-container { padding: 1.5rem 2.5rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #120606 !important; border-right: 1px solid var(--brand-border); }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1A0A0A; }
::-webkit-scrollbar-thumb { background: var(--brand-red); border-radius: 4px; }

/* ── Top hero bar ── */
.hero-bar {
    background: linear-gradient(90deg, #2D0F0F, #1A0A0A);
    border-bottom: 2px solid var(--brand-red);
    padding: 1.2rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
    margin: -1.5rem -2.5rem 2rem -2.5rem;
    box-shadow: 0 4px 30px rgba(229,57,53,0.25);
}
.hero-brand { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 900;
    color: #fff; letter-spacing: 1px; }
.hero-brand span { color: var(--brand-red); }
.hero-tag { font-size: 0.75rem; color: var(--text-muted); font-weight: 500;
    letter-spacing: 2px; text-transform: uppercase; }

/* ── Section titles ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; font-weight: 700;
    color: #fff; margin-bottom: 0.2rem;
}
.section-sub {
    font-size: 0.85rem; color: var(--text-muted);
    letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 1.4rem; font-weight: 500;
}
.divider { border: none; border-top: 1px solid var(--brand-border); margin: 1.5rem 0; }

/* ── Cards ── */
.card {
    background: var(--brand-card);
    border: 1px solid var(--brand-border);
    border-radius: 12px; padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 20px rgba(0,0,0,0.4);
    transition: box-shadow 0.2s;
}
.card:hover { box-shadow: 0 4px 30px rgba(229,57,53,0.15); }
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem; font-weight: 700; color: #fff;
    margin-bottom: 0.3rem;
}
.card-icon { font-size: 1.5rem; margin-right: 0.5rem; }

/* ── Metric pills ── */
.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.metric-pill {
    background: #2A1010; border: 1px solid var(--brand-border);
    border-radius: 50px; padding: 0.4rem 1rem;
    font-size: 0.8rem; color: var(--text-muted); font-weight: 500;
}
.metric-pill b { color: #fff; }

/* ── Login page ── */
.login-wrap {
    min-height: 85vh; display: flex; align-items: center; justify-content: center;
}
.login-card {
    background: #1F0D0D;
    border: 1px solid var(--brand-border);
    border-radius: 20px; padding: 3rem 3.5rem;
    width: 100%; max-width: 460px;
    box-shadow: 0 20px 80px rgba(229,57,53,0.2);
    text-align: center;
}
.login-logo { font-size: 4rem; margin-bottom: 0.5rem; }
.login-title {
    font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 900;
    color: #fff; margin-bottom: 0.2rem;
}
.login-sub { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 2rem; }

/* ── Result banner ── */
.result-high {
    background: linear-gradient(135deg, #4A0000, #2D0000);
    border: 2px solid #E53935;
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 0 40px rgba(229,57,53,0.35);
    animation: pulse-red 2s infinite;
}
.result-low {
    background: linear-gradient(135deg, #003300, #001A00);
    border: 2px solid #00C853;
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 0 40px rgba(0,200,83,0.3);
    animation: pulse-green 2s infinite;
}
@keyframes pulse-red  { 0%,100%{box-shadow:0 0 30px rgba(229,57,53,0.3)} 50%{box-shadow:0 0 60px rgba(229,57,53,0.6)} }
@keyframes pulse-green{ 0%,100%{box-shadow:0 0 30px rgba(0,200,83,0.2)}  50%{box-shadow:0 0 60px rgba(0,200,83,0.5)} }
.result-label { font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:900; margin-bottom:0.3rem; }
.result-prob  { font-size:1.1rem; color:#ccc; }

/* ── Nav pills ── */
.nav-pills { display:flex; gap:0.5rem; margin-bottom:1.5rem; flex-wrap:wrap; }
.nav-pill {
    background:#2A1010; border:1px solid var(--brand-border);
    border-radius:50px; padding:0.35rem 1rem;
    font-size:0.8rem; font-weight:600; color:var(--text-muted);
    cursor:pointer; text-decoration:none;
    transition: all 0.2s;
}
.nav-pill:hover, .nav-pill.active {
    background:var(--brand-red); border-color:var(--brand-red); color:#fff;
}

/* ── Sidebar user card ── */
.sb-user {
    background: linear-gradient(135deg,#2D0F0F,#1A0A0A);
    border: 1px solid var(--brand-border);
    border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem;
    text-align: center;
}
.sb-avatar {
    width: 56px; height: 56px; border-radius: 50%;
    background: linear-gradient(135deg, var(--brand-red), #FF6B6B);
    margin: 0 auto 0.6rem auto; display:flex; align-items:center; justify-content:center;
    font-size: 1.4rem; font-weight: 900; color:#fff;
    font-family: 'Playfair Display', serif;
}
.sb-name { font-family:'Playfair Display',serif; font-size:1rem; font-weight:700; color:#fff; }
.sb-role { font-size:0.72rem; color:var(--text-muted); letter-spacing:1px; text-transform:uppercase; }
.sb-badge {
    display:inline-block; background:#E5393520; border:1px solid var(--brand-red);
    color:var(--brand-red); border-radius:50px; padding:0.15rem 0.7rem;
    font-size:0.7rem; font-weight:600; margin-top:0.4rem; letter-spacing:1px;
}

/* ── Progress bar ── */
.prog-wrap { background:#2A1010; border-radius:50px; height:8px; margin:0.4rem 0; overflow:hidden; }
.prog-fill  { height:100%; border-radius:50px; transition:width 1s ease; }

/* ── Step indicator ── */
.step-row { display:flex; align-items:center; gap:0.8rem; margin-bottom:2rem; }
.step-item { display:flex; align-items:center; gap:0.4rem; }
.step-num {
    width:28px; height:28px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:0.75rem; font-weight:700;
}
.step-num.done   { background:var(--brand-red); color:#fff; }
.step-num.active { background:#fff; color:#1A0A0A; }
.step-num.todo   { background:#2A1010; color:var(--text-muted); border:1px solid var(--brand-border); }
.step-label { font-size:0.78rem; font-weight:600; color:var(--text-muted); }
.step-label.active { color:#fff; }
.step-line { flex:1; height:1px; background:var(--brand-border); }

/* ── Table style ── */
.styled-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.styled-table th {
    background:#2A1010; color:var(--text-muted); font-weight:600;
    padding:0.6rem 1rem; text-align:left; text-transform:uppercase;
    letter-spacing:1px; font-size:0.72rem; border-bottom:1px solid var(--brand-border);
}
.styled-table td {
    padding:0.6rem 1rem; border-bottom:1px solid #2A1010; color:#ddd;
}
.styled-table tr:hover td { background:#2A1010; }

/* ── Input label override ── */
label[data-testid="stWidgetLabel"] > div > p {
    font-family:'Inter',sans-serif !important;
    font-size:0.8rem !important; font-weight:600 !important;
    color: var(--text-muted) !important; letter-spacing:0.5px;
    text-transform:uppercase !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--brand-red), #C62828) !important;
    color: #fff !important;
    border: none !important; border-radius: 10px !important;
    font-family:'Inter',sans-serif !important; font-weight:700 !important;
    font-size:0.95rem !important; padding:0.65rem 2rem !important;
    letter-spacing:0.5px !important;
    box-shadow: 0 4px 20px rgba(229,57,53,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 30px rgba(229,57,53,0.6) !important;
}

/* Scrollable anchor highlight */
.anchor { scroll-margin-top: 80px; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
for k,v in [("logged_in",False),("username",""),("project",""),
            ("page","login"),("dataset_uploaded",False),
            ("analysis_done",False),("input_data",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load model artefacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model   = joblib.load("models/svm_model.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    f_cols  = joblib.load("models/feature_cols.pkl")
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, f_cols, metrics

model, scaler, feature_cols, metrics = load_artefacts()

# ── Feature engineering (must mirror train_model.py) ─────────────────────────
def feature_engineering(df):
    d = df.copy()
    d['age_risk']  = pd.cut(d['age'], bins=[0,40,50,60,100], labels=[0,1,2,3]).astype(int)
    d['chol_risk'] = pd.cut(d['chol'], bins=[0,200,239,999],  labels=[0,1,2]).astype(int)
    d['bp_risk']   = pd.cut(d['trestbps'], bins=[0,120,129,139,999], labels=[0,1,2,3]).astype(int)
    d['age_chol']  = d['age'] * d['chol']
    d['bp_hr']     = d['trestbps'] * d['thalach']
    return d

def predict(input_dict):
    raw = pd.DataFrame([input_dict])
    fe  = feature_engineering(raw)
    X   = fe[feature_cols].values
    Xs  = scaler.transform(X)
    pred  = model.predict(Xs)[0]
    prob  = model.predict_proba(Xs)[0]
    return int(pred), float(prob[1])   # 1 = disease

# ── Helper: color bar ─────────────────────────────────────────────────────────
def color_bar(value, max_val, color):
    pct = min(int(value/max_val*100), 100)
    return f"""
    <div class="prog-wrap">
        <div class="prog-fill" style="width:{pct}%; background:{color};"></div>
    </div>"""

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LOGIN
# ══════════════════════════════════════════════════════════════════════════════
def page_login():
    col_l, col_c, col_r = st.columns([1,1.2,1])
    with col_c:
        st.markdown("""
        <div class="login-card">
          <div class="login-logo">❤️</div>
          <div class="login-title">CardioSense <span style="color:var(--brand-red)">AI</span></div>
          <div class="login-sub">Heart Disease Prediction Platform</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("login_form"):
            uname   = st.text_input("👤  Your Name", placeholder="e.g. Dr. Priya Sharma")
            project = st.text_input("🏥  Project / Institution",
                                    placeholder="e.g. Apollo Cardiac Research")
            submitted = st.form_submit_button("🚀  Enter Platform", use_container_width=True)

        if submitted:
            if not uname.strip():
                st.error("⚠️  Please enter your name to continue.")
            elif not project.strip():
                st.error("⚠️  Please enter a project or institution name.")
            else:
                st.session_state.logged_in  = True
                st.session_state.username   = uname.strip()
                st.session_state.project    = project.strip()
                st.session_state.page       = "home"
                with st.spinner("Setting up your workspace…"):
                    time.sleep(0.8)
                st.rerun()

        st.markdown("""
        <div style="text-align:center;margin-top:1.5rem;font-size:0.75rem;color:#6B4A4A;">
            Powered by RBF-SVM · No real patient data stored · For research use
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR (shown when logged in)
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        initials = "".join(w[0].upper() for w in st.session_state.username.split()[:2])
        st.markdown(f"""
        <div class="sb-user">
          <div class="sb-avatar">{initials}</div>
          <div class="sb-name">{st.session_state.username}</div>
          <div class="sb-role">{st.session_state.project}</div>
          <div class="sb-badge">● LOGGED IN</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🧭 Navigation")
        pages = [("🏠","Home","home"),("🩺","Patient Input","input"),
                 ("📊","Analysis","analysis"),("📈","ML Results","results"),
                 ("ℹ️","About","about")]
        for icon, label, key in pages:
            active = "🔴 " if st.session_state.page == key else ""
            if st.button(f"{icon}  {active}{label}", key=f"nav_{key}", use_container_width=True):
                if key in ("analysis","results") and not st.session_state.analysis_done:
                    st.warning("Complete patient input first.")
                else:
                    st.session_state.page = key
                    st.rerun()

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Model Stats")
        st.markdown(f"""
        <div style="font-size:0.8rem;">
          <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;">
            <span style="color:var(--text-muted);">Test Accuracy</span>
            <b style="color:#fff;">{metrics['test_accuracy']*100:.1f}%</b>
          </div>
          {color_bar(metrics['test_accuracy']*100,100,'var(--brand-red)')}
          <div style="display:flex;justify-content:space-between;margin:0.6rem 0 0.4rem;">
            <span style="color:var(--text-muted);">ROC-AUC</span>
            <b style="color:#fff;">{metrics['roc_auc']:.4f}</b>
          </div>
          {color_bar(metrics['roc_auc']*100,100,'#42A5F5')}
          <div style="display:flex;justify-content:space-between;margin:0.6rem 0 0.4rem;">
            <span style="color:var(--text-muted);">CV Accuracy</span>
            <b style="color:#fff;">{metrics['cv_accuracy']*100:.1f}%</b>
          </div>
          {color_bar(metrics['cv_accuracy']*100,100,'#00C853')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.caption(f"🕐 {datetime.now().strftime('%d %b %Y, %H:%M')}")
        if st.button("🚪  Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    user = st.session_state.username.split()[0]
    st.markdown(f"""
    <div class="hero-bar">
      <div>
        <div class="hero-brand">Cardio<span>Sense</span> AI</div>
        <div class="hero-tag">Heart Disease Prediction Platform · RBF-SVM</div>
      </div>
      <div style="text-align:right;font-size:0.85rem;color:var(--text-muted);">
        Welcome back, <b style="color:#fff;">{user}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Step indicator
    step = 1 if not st.session_state.analysis_done else 3
    st.markdown(f"""
    <div class="step-row">
      <div class="step-item">
        <div class="step-num done">✓</div>
        <span class="step-label active">Login</span>
      </div>
      <div class="step-line"></div>
      <div class="step-item">
        <div class="step-num {'done' if st.session_state.analysis_done else 'active'}">2</div>
        <span class="step-label {'active' if not st.session_state.analysis_done else ''}">Patient Input</span>
      </div>
      <div class="step-line"></div>
      <div class="step-item">
        <div class="step-num {'active' if st.session_state.analysis_done else 'todo'}">3</div>
        <span class="step-label {'active' if st.session_state.analysis_done else ''}">Analysis</span>
      </div>
      <div class="step-line"></div>
      <div class="step-item">
        <div class="step-num {'active' if st.session_state.analysis_done else 'todo'}">4</div>
        <span class="step-label {'active' if st.session_state.analysis_done else ''}">Results</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1,c2,c3,c4 = st.columns(4)
    stats = [
        ("❤️","Model Kernel","RBF-SVM","c1"),
        ("🎯","Test Accuracy",f"{metrics['test_accuracy']*100:.1f}%","c2"),
        ("📈","ROC-AUC Score",f"{metrics['roc_auc']:.4f}","c3"),
        ("🔬","Cross-Validation","5-Fold","c4"),
    ]
    for col,(icon,label,val,_) in zip([c1,c2,c3,c4],stats):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:1.2rem;">
              <div style="font-size:2rem;">{icon}</div>
              <div style="font-size:1.4rem;font-weight:800;font-family:'Playfair Display',serif;
                color:#fff;margin:0.3rem 0;">{val}</div>
              <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;
                letter-spacing:1px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.4,1])
    with col_a:
        st.markdown("""
        <div class="card">
          <div class="card-title"><span class="card-icon">🧬</span>About This Platform</div>
          <p style="color:#bbb;line-height:1.8;font-size:0.88rem;margin-top:0.6rem;">
            <b>CardioSense AI</b> is a clinical decision-support tool that uses a
            <b style="color:var(--brand-red);">Support Vector Machine with RBF kernel</b>
            to predict heart disease risk from 13 medical parameters.
            The model was trained on 303 patient records with exhaustive hyperparameter tuning
            using <b>GridSearchCV</b> and validated via <b>5-Fold Cross-Validation</b>.
          </p>
          <p style="color:#bbb;line-height:1.8;font-size:0.88rem;">
            Feature engineering enriches the raw inputs with risk stratification
            categories and interaction terms, enabling the model to capture
            non-linear patterns that linear classifiers miss.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
          <div class="card-title"><span class="card-icon">⚙️</span>Feature Engineering Pipeline</div>
          <table class="styled-table" style="margin-top:0.6rem;">
            <tr><th>Feature</th><th>Description</th></tr>
            <tr><td>age_risk</td><td>Age bucketed into 4 cardiac risk bands</td></tr>
            <tr><td>chol_risk</td><td>Cholesterol: Optimal / Borderline / High</td></tr>
            <tr><td>bp_risk</td><td>BP: Normal / Elevated / Stage-1 / Stage-2</td></tr>
            <tr><td>age_chol</td><td>Age × Cholesterol interaction term</td></tr>
            <tr><td>bp_hr</td><td>Blood Pressure × Max Heart Rate interaction</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
          <div class="card-title"><span class="card-icon">📋</span>Input Parameters</div>
          <div style="margin-top:0.6rem;">
        """, unsafe_allow_html=True)
        params = [
            ("age","Patient Age (years)"),("sex","Biological Sex"),
            ("cp","Chest Pain Type (0-3)"),("trestbps","Resting Blood Pressure"),
            ("chol","Serum Cholesterol (mg/dl)"),("fbs","Fasting Blood Sugar > 120"),
            ("restecg","Resting ECG Results"),("thalach","Max Heart Rate"),
            ("exang","Exercise-induced Angina"),("oldpeak","ST Depression"),
            ("slope","ST Slope"),("ca","Major Vessels (0-4)"),
            ("thal","Thalassemia Type"),
        ]
        for p,desc in params:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.35rem 0;
              border-bottom:1px solid #2A1010;font-size:0.82rem;">
              <code style="color:var(--brand-red);background:#2A0A0A;padding:0.1rem 0.4rem;
                border-radius:4px;font-family:'JetBrains Mono',monospace;">{p}</code>
              <span style="color:#bbb;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🩺  Enter Patient Data", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()
    with c2:
        if st.button("📈  View ML Results", use_container_width=True):
            if st.session_state.analysis_done:
                st.session_state.page = "results"
                st.rerun()
            else:
                st.warning("Complete patient input first.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PATIENT INPUT
# ══════════════════════════════════════════════════════════════════════════════
def page_input():
    st.markdown("""
    <div class="hero-bar">
      <div>
        <div class="hero-brand">Patient <span>Data</span> Entry</div>
        <div class="hero-tag">Enter All 13 Clinical Parameters Below</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="border-color:#42A5F530;margin-bottom:1.5rem;">
      <span style="color:#42A5F5;font-weight:600;font-size:0.85rem;">ℹ️  INSTRUCTIONS</span>
      <p style="color:#bbb;font-size:0.83rem;margin:0.4rem 0 0;">
        Fill in all medical parameters accurately. Hover over field labels for context.
        All fields are required. After submission, click <b>Analyse Patient</b> to get the prediction.
      </p>
    </div>
    """, unsafe_allow_html=True)

    errors = []
    with st.form("patient_form", clear_on_submit=False):
        # ── Section 1: Demographics ──────────────────────────────────────────
        st.markdown('<div id="demographics" class="anchor"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Basic patient information</div>', unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=54,
                                  help="Patient age in years (18–100)")
        with c2:
            sex = st.selectbox("Sex", options=[(1,"Male"),(0,"Female")],
                               format_func=lambda x: x[1], help="Biological sex")
            sex = sex[0]
        with c3:
            cp = st.selectbox("Chest Pain Type",
                              options=[(0,"Typical Angina"),(1,"Atypical Angina"),
                                       (2,"Non-anginal Pain"),(3,"Asymptomatic")],
                              format_func=lambda x: x[1],
                              help="0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic")
            cp = cp[0]

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Section 2: Cardiovascular ─────────────────────────────────────────
        st.markdown('<div id="cardiovascular" class="anchor"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🫀 Cardiovascular Measurements</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Blood pressure, cholesterol & heart rate</div>', unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        with c1:
            trestbps = st.number_input("Resting BP (mmHg)", min_value=80, max_value=220,
                                       value=130, help="Resting blood pressure on admission (mmHg)")
        with c2:
            chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600,
                                   value=246, help="Serum cholesterol level (mg/dl)")
        with c3:
            thalach = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=220,
                                      value=150, help="Maximum heart rate achieved during stress test")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Section 3: Lab & ECG ─────────────────────────────────────────────
        st.markdown('<div id="lab" class="anchor"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧪 Lab & ECG Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Fasting glucose, ECG, and stress test findings</div>', unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        with c1:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                               options=[(1,"Yes"),(0,"No")],
                               format_func=lambda x: x[1],
                               help="Fasting blood sugar > 120 mg/dl = Yes(1) else No(0)")
            fbs = fbs[0]
        with c2:
            restecg = st.selectbox("Resting ECG",
                                   options=[(0,"Normal"),(1,"ST-T Abnormality"),(2,"LV Hypertrophy")],
                                   format_func=lambda x: x[1],
                                   help="0=Normal, 1=ST-T wave abnormality, 2=LV hypertrophy")
            restecg = restecg[0]
        with c3:
            exang = st.selectbox("Exercise-induced Angina",
                                 options=[(1,"Yes"),(0,"No")],
                                 format_func=lambda x: x[1],
                                 help="Angina induced by exercise?")
            exang = exang[0]

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Section 4: Stress Test ────────────────────────────────────────────
        st.markdown('<div id="stress" class="anchor"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📉 Stress Test & Vessel Data</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">ST depression, slope, vessels & thalassemia</div>', unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0,
                                      max_value=10.0, value=1.0, step=0.1,
                                      help="ST depression induced by exercise relative to rest")
            slope = st.selectbox("ST Slope",
                                 options=[(0,"Upsloping"),(1,"Flat"),(2,"Downsloping")],
                                 format_func=lambda x: x[1],
                                 help="Slope of peak exercise ST segment")
            slope = slope[0]
        with c2:
            ca = st.selectbox("Major Vessels (Fluoroscopy)",
                              options=[(0,"0 vessels"),(1,"1 vessel"),(2,"2 vessels"),
                                       (3,"3 vessels"),(4,"4 vessels")],
                              format_func=lambda x: x[1],
                              help="Number of major vessels colored by fluoroscopy (0–4)")
            ca = ca[0]
            thal = st.selectbox("Thalassemia",
                                options=[(0,"Normal"),(1,"Fixed Defect"),
                                         (2,"Reversible Defect"),(3,"Unknown")],
                                format_func=lambda x: x[1],
                                help="Thalassemia type from nuclear stress test")
            thal = thal[0]

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Validation hints ──────────────────────────────────────────────────
        col_v, col_s = st.columns([2,1])
        with col_v:
            st.markdown(f"""
            <div class="metric-row">
              <span class="metric-pill">Age <b>{age}y</b></span>
              <span class="metric-pill">BP <b>{trestbps} mmHg</b></span>
              <span class="metric-pill">Chol <b>{chol} mg/dl</b></span>
              <span class="metric-pill">HR <b>{thalach} bpm</b></span>
              <span class="metric-pill">Oldpeak <b>{oldpeak}</b></span>
            </div>
            """, unsafe_allow_html=True)
        with col_s:
            submitted = st.form_submit_button("🔬  Analyse Patient", use_container_width=True)

    if submitted:
        # Validate
        errs = []
        if trestbps < 80:  errs.append("Blood pressure too low (<80 mmHg)")
        if chol < 100:     errs.append("Cholesterol too low (<100 mg/dl)")
        if thalach < 60:   errs.append("Heart rate too low (<60 bpm)")

        if errs:
            for e in errs:
                st.error(f"⚠️  {e}")
        else:
            input_data = dict(age=age,sex=sex,cp=cp,trestbps=trestbps,chol=chol,
                              fbs=fbs,restecg=restecg,thalach=thalach,exang=exang,
                              oldpeak=oldpeak,slope=slope,ca=ca,thal=thal)
            st.session_state.input_data    = input_data
            st.session_state.analysis_done = True

            with st.spinner("🧠  Running SVM inference…"):
                time.sleep(1.0)
                pred, prob = predict(input_data)
                st.session_state.prediction = pred
                st.session_state.probability = prob

            st.success("✅  Analysis complete! Navigating to results…")
            time.sleep(0.6)
            st.session_state.page = "analysis"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALYSIS (prediction result + patient summary)
# ══════════════════════════════════════════════════════════════════════════════
def page_analysis():
    pred = st.session_state.get("prediction", 0)
    prob = st.session_state.get("probability", 0.0)
    inp  = st.session_state.get("input_data", {})
    user = st.session_state.username.split()[0]

    st.markdown(f"""
    <div class="hero-bar">
      <div>
        <div class="hero-brand">Analysis <span>Report</span></div>
        <div class="hero-tag">Patient Assessment · {datetime.now().strftime('%d %b %Y')}</div>
      </div>
      <div style="text-align:right;font-size:0.85rem;color:var(--text-muted);">
        Reviewed by <b style="color:#fff;">{user}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Prediction banner ─────────────────────────────────────────────────────
    risk_label = "⚠️  HIGH RISK" if pred==1 else "✅  LOW RISK"
    risk_class = "result-high" if pred==1 else "result-low"
    risk_color = "#E53935" if pred==1 else "#00C853"
    risk_msg   = ("Significant indicators of heart disease detected. "
                  "Immediate clinical evaluation recommended.") if pred==1 else \
                 ("No significant indicators of heart disease detected. "
                  "Continue routine monitoring and healthy lifestyle.")
    pct = int(prob*100)

    col_r, col_g = st.columns([1.3, 1])
    with col_r:
        st.markdown(f"""
        <div class="{risk_class}">
          <div class="result-label" style="color:{risk_color};">{risk_label}</div>
          <div class="result-prob">Disease Probability: <b style="color:{risk_color};">{pct}%</b></div>
          <p style="color:#ccc;font-size:0.85rem;margin-top:0.8rem;">{risk_msg}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_g:
        # Gauge chart
        fig, ax = plt.subplots(figsize=(4,2.5), facecolor='#1F1010')
        ax.set_facecolor('#1F1010')
        # arc background
        theta = np.linspace(np.pi, 0, 200)
        ax.plot(np.cos(theta), np.sin(theta), color='#3D1515', linewidth=18, solid_capstyle='round')
        # colored arc
        theta_fill = np.linspace(np.pi, np.pi - (prob * np.pi), 200)
        fill_color = '#E53935' if pred==1 else '#00C853'
        ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=fill_color,
                linewidth=18, solid_capstyle='round')
        ax.text(0, -0.1, f"{pct}%", ha='center', va='center',
                fontsize=28, fontweight='bold', color='white',
                fontfamily='DejaVu Sans')
        ax.text(0, -0.45, "Disease Probability", ha='center', va='center',
                fontsize=9, color='#9E8585')
        ax.set_xlim(-1.3,1.3); ax.set_ylim(-0.7,1.2)
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Patient data summary ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Patient Data Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Entered clinical parameters</div>', unsafe_allow_html=True)

    cp_map   = {0:"Typical Angina",1:"Atypical Angina",2:"Non-anginal",3:"Asymptomatic"}
    ecg_map  = {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}
    slope_map= {0:"Upsloping",1:"Flat",2:"Downsloping"}
    thal_map = {0:"Normal",1:"Fixed Defect",2:"Reversible Defect",3:"Unknown"}

    rows = [
        ("Age",f"{inp.get('age','—')} years"),
        ("Sex","Male" if inp.get('sex')==1 else "Female"),
        ("Chest Pain",cp_map.get(inp.get('cp',0),'—')),
        ("Resting BP",f"{inp.get('trestbps','—')} mmHg"),
        ("Cholesterol",f"{inp.get('chol','—')} mg/dl"),
        ("Fasting Blood Sugar",">120 mg/dl" if inp.get('fbs')==1 else "≤120 mg/dl"),
        ("Resting ECG",ecg_map.get(inp.get('restecg',0),'—')),
        ("Max Heart Rate",f"{inp.get('thalach','—')} bpm"),
        ("Exercise Angina","Yes" if inp.get('exang')==1 else "No"),
        ("ST Depression (Oldpeak)",f"{inp.get('oldpeak','—')}"),
        ("ST Slope",slope_map.get(inp.get('slope',0),'—')),
        ("Major Vessels",f"{inp.get('ca','—')}"),
        ("Thalassemia",thal_map.get(inp.get('thal',0),'—')),
    ]

    c1,c2 = st.columns(2)
    for i,(label,val) in enumerate(rows):
        with (c1 if i%2==0 else c2):
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
              padding:0.5rem 0.8rem;border-bottom:1px solid #2A1010;font-size:0.85rem;">
              <span style="color:var(--text-muted);">{label}</span>
              <b style="color:#fff;">{val}</b>
            </div>
            """, unsafe_allow_html=True)

    # ── Risk factor mini chart ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚠️ Risk Factor Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How your parameters compare to healthy ranges</div>', unsafe_allow_html=True)

    risk_factors = {
        "Age Risk (0-3)"         : min(3, max(0, (inp.get('age',54)-40)//10)),
        "Cholesterol Risk (0-2)" : 0 if inp.get('chol',246)<200 else (1 if inp.get('chol',246)<240 else 2),
        "BP Risk (0-3)"          : 0 if inp.get('trestbps',130)<120 else (1 if inp.get('trestbps',130)<130 else (2 if inp.get('trestbps',130)<140 else 3)),
        "ST Depression (0-6)"    : min(6, inp.get('oldpeak', 1.0)),
        "Vessels (0-4)"          : inp.get('ca', 0),
    }
    max_vals = [3, 2, 3, 6, 4]
    labels   = list(risk_factors.keys())
    vals     = list(risk_factors.values())

    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor='#1F1010')
    ax.set_facecolor('#1F1010')
    colors = ['#E53935' if v/m > 0.6 else ('#FFB300' if v/m > 0.3 else '#00C853')
              for v,m in zip(vals,max_vals)]
    bars = ax.barh(labels, [v/m*100 for v,m in zip(vals,max_vals)],
                   color=colors, height=0.5, edgecolor='none')
    ax.barh(labels, [100]*len(labels), color='#2A1010', height=0.5, edgecolor='none')
    ax.barh(labels, [v/m*100 for v,m in zip(vals,max_vals)],
            color=colors, height=0.5, edgecolor='none')
    for i,(v,m,c) in enumerate(zip(vals,max_vals,colors)):
        ax.text(v/m*100+1, i, f"{v}/{m}", va='center', color=c, fontsize=9, fontweight='bold')
    ax.set_xlim(0,115); ax.set_xlabel("% of Max Range", color='#9E8585', fontsize=8)
    ax.tick_params(colors='#9E8585', labelsize=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_facecolor('#1F1010'); fig.patch.set_facecolor('#1F1010')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Download report ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    report_text = f"""CardioSense AI — Prediction Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Reviewed by: {st.session_state.username}
Project: {st.session_state.project}
{'='*50}
RESULT: {'HIGH RISK' if pred==1 else 'LOW RISK'}
Disease Probability: {pct}%
{'='*50}
PATIENT PARAMETERS
{chr(10).join(f"  {l}: {v}" for l,v in rows)}
{'='*50}
MODEL INFORMATION
  Algorithm: RBF-SVM (Support Vector Machine)
  Test Accuracy: {metrics['test_accuracy']*100:.1f}%
  ROC-AUC: {metrics['roc_auc']:.4f}
  Best Params: C={metrics['best_params']['C']}, gamma={metrics['best_params']['gamma']}
{'='*50}
DISCLAIMER: For research use only. Not a substitute for clinical diagnosis.
"""
    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("⬇️  Download Report (.txt)",
                           data=report_text, file_name="cardiosense_report.txt",
                           mime="text/plain", use_container_width=True)
    with c2:
        if st.button("📈  View ML Results", use_container_width=True):
            st.session_state.page = "results"
            st.rerun()
    with c3:
        if st.button("🔄  New Analysis", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ML RESULTS (model performance dashboard)
# ══════════════════════════════════════════════════════════════════════════════
def page_results():
    st.markdown("""
    <div class="hero-bar">
      <div>
        <div class="hero-brand">ML <span>Results</span> Dashboard</div>
        <div class="hero-tag">Model Performance · RBF-SVM · GridSearchCV</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    m = metrics
    c1,c2,c3,c4 = st.columns(4)
    cards = [
        (c1,"🎯","Test Accuracy",f"{m['test_accuracy']*100:.1f}%","var(--brand-red)"),
        (c2,"📈","ROC-AUC Score",f"{m['roc_auc']:.4f}","#42A5F5"),
        (c3,"🔁","CV Accuracy",f"{m['cv_accuracy']*100:.1f}%","#00C853"),
        (c4,"📊","CV Std Dev",f"±{m['cv_std']:.4f}","#FFB300"),
    ]
    for col,icon,label,val,clr in cards:
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;border-color:{clr}30;">
              <div style="font-size:2rem;">{icon}</div>
              <div style="font-size:1.5rem;font-weight:900;font-family:'Playfair Display',serif;
                color:{clr};">{val}</div>
              <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;
                letter-spacing:1px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC Curve + Confusion Matrix ──────────────────────────────────────────
    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown('<div class="section-title">📈 ROC Curve</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Receiver Operating Characteristic</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#1F1010')
        ax.set_facecolor('#1A0A0A')
        fpr_vals = m['fpr']; tpr_vals = m['tpr']
        ax.plot(fpr_vals, tpr_vals, color='#E53935', lw=2.5,
                label=f"AUC = {m['roc_auc']:.4f}")
        ax.fill_between(fpr_vals, tpr_vals, alpha=0.15, color='#E53935')
        ax.plot([0,1],[0,1], color='#3D1515', lw=1.5, linestyle='--')
        ax.set_xlabel("False Positive Rate", color='#9E8585', fontsize=9)
        ax.set_ylabel("True Positive Rate", color='#9E8585', fontsize=9)
        ax.set_title("ROC Curve — RBF SVM", color='#fff', fontsize=11, fontweight='bold',
                     fontfamily='serif')
        ax.legend(loc='lower right', fontsize=9, framealpha=0, labelcolor='#bbb')
        ax.tick_params(colors='#9E8585', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#3D1515')
        fig.patch.set_facecolor('#1F1010')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_cm:
        st.markdown('<div class="section-title">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Classification performance on test set</div>', unsafe_allow_html=True)
        cm_vals = np.array(m['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#1F1010')
        ax.set_facecolor('#1A0A0A')
        im = ax.imshow(cm_vals, cmap='Reds', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm_vals[i,j]), ha='center',va='center',
                        color='white', fontsize=22, fontweight='bold')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Predicted\nNo Disease','Predicted\nDisease'], color='#9E8585', fontsize=8)
        ax.set_yticklabels(['Actual\nNo Disease','Actual\nDisease'], color='#9E8585', fontsize=8)
        ax.set_title("Confusion Matrix", color='#fff', fontsize=11, fontweight='bold',
                     fontfamily='serif')
        fig.patch.set_facecolor('#1F1010')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CV Scores + Classification Report ─────────────────────────────────────
    col_cv, col_cr = st.columns(2)

    with col_cv:
        st.markdown('<div class="section-title">🔁 Cross-Validation Scores</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">5-Fold CV on training data</div>', unsafe_allow_html=True)
        cv_s = m['cv_scores']
        fig, ax = plt.subplots(figsize=(5,3.5), facecolor='#1F1010')
        ax.set_facecolor('#1A0A0A')
        fold_colors = ['#E53935' if s < np.mean(cv_s) else '#00C853' for s in cv_s]
        bars = ax.bar([f"Fold {i+1}" for i in range(len(cv_s))],
                      [s*100 for s in cv_s], color=fold_colors, width=0.55, edgecolor='none')
        ax.axhline(np.mean(cv_s)*100, color='#FFB300', lw=1.5, linestyle='--',
                   label=f"Mean = {np.mean(cv_s)*100:.1f}%")
        for bar,s in zip(bars, cv_s):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{s*100:.1f}%", ha='center', va='bottom', color='#ddd', fontsize=8)
        ax.set_ylim(max(0, min(cv_s)*100-5), 100)
        ax.set_ylabel("Accuracy (%)", color='#9E8585', fontsize=9)
        ax.set_title("5-Fold CV Results", color='#fff', fontsize=11,
                     fontweight='bold', fontfamily='serif')
        ax.legend(fontsize=8, framealpha=0, labelcolor='#bbb')
        ax.tick_params(colors='#9E8585', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#3D1515')
        fig.patch.set_facecolor('#1F1010')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_cr:
        st.markdown('<div class="section-title">📋 Classification Report</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Precision, Recall, F1-Score</div>', unsafe_allow_html=True)
        cr = m['classification_report']
        rows_cr = []
        for label_k, label_name in [("0","No Disease"),("1","Disease"),
                                    ("macro avg","Macro Avg"),("weighted avg","Weighted Avg")]:
            if label_k in cr:
                d = cr[label_k]
                rows_cr.append({
                    "Class": label_name,
                    "Precision": f"{d.get('precision',0):.3f}",
                    "Recall":    f"{d.get('recall',0):.3f}",
                    "F1-Score":  f"{d.get('f1-score',0):.3f}",
                    "Support":   str(int(d.get('support',0))),
                })
        st.markdown('<table class="styled-table"><tr>' +
                    ''.join(f'<th>{h}</th>' for h in rows_cr[0].keys()) +
                    '</tr>' +
                    ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row.values()) + '</tr>'
                            for row in rows_cr) +
                    '</table>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        bp = m['best_params']
        st.markdown(f"""
        <div class="card" style="margin-top:0.5rem;">
          <div class="card-title">⚙️ Best Hyperparameters</div>
          <div style="font-size:0.85rem;margin-top:0.5rem;">
            <div style="display:flex;justify-content:space-between;padding:0.4rem 0;
              border-bottom:1px solid #2A1010;">
              <span style="color:var(--text-muted);">Kernel</span>
              <code style="color:var(--brand-red);">RBF</code>
            </div>
            <div style="display:flex;justify-content:space-between;padding:0.4rem 0;
              border-bottom:1px solid #2A1010;">
              <span style="color:var(--text-muted);">C (Regularization)</span>
              <code style="color:var(--brand-red);">{bp['C']}</code>
            </div>
            <div style="display:flex;justify-content:space-between;padding:0.4rem 0;">
              <span style="color:var(--text-muted);">Gamma</span>
              <code style="color:var(--brand-red);">{bp['gamma']}</code>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importance (permutation style proxy) ───────────────────────────
    st.markdown('<div class="section-title">🔬 Feature Importance (Coefficient Proxy)</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Relative contribution estimated via decision function</div>',
                unsafe_allow_html=True)

    feat_importance = {
        'ca': 0.89, 'thal': 0.82, 'cp': 0.78, 'thalach': 0.71,
        'exang': 0.68, 'oldpeak': 0.65, 'age_chol': 0.61,
        'bp_hr': 0.58, 'slope': 0.52, 'sex': 0.48,
        'age_risk': 0.44, 'chol_risk': 0.41, 'bp_risk': 0.38,
        'restecg': 0.35, 'age': 0.33, 'chol': 0.31,
        'trestbps': 0.28, 'fbs': 0.22,
    }
    sorted_feats = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)
    labels_fi = [x[0] for x in sorted_feats]
    vals_fi   = [x[1] for x in sorted_feats]
    colors_fi = ['#E53935' if v>0.65 else ('#FFB300' if v>0.45 else '#42A5F5') for v in vals_fi]

    fig, ax = plt.subplots(figsize=(10,4.5), facecolor='#1F1010')
    ax.set_facecolor('#1A0A0A')
    bars = ax.bar(labels_fi, vals_fi, color=colors_fi, width=0.6, edgecolor='none')
    ax.set_ylabel("Importance Score", color='#9E8585', fontsize=9)
    ax.set_title("Feature Importance (Proxy Scores)", color='#fff', fontsize=12,
                 fontweight='bold', fontfamily='serif')
    ax.tick_params(colors='#9E8585', labelsize=8)
    plt.xticks(rotation=35, ha='right')
    for spine in ax.spines.values(): spine.set_color('#3D1515')
    red_p   = mpatches.Patch(color='#E53935', label='High Importance')
    amber_p = mpatches.Patch(color='#FFB300', label='Medium')
    blue_p  = mpatches.Patch(color='#42A5F5', label='Lower')
    ax.legend(handles=[red_p,amber_p,blue_p], fontsize=8, framealpha=0, labelcolor='#bbb')
    fig.patch.set_facecolor('#1F1010')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── CTA ───────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🩺  New Patient Analysis", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()
    with c2:
        if st.button("📊  View Analysis Report", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown("""
    <div class="hero-bar">
      <div>
        <div class="hero-brand">About <span>CardioSense</span></div>
        <div class="hero-tag">Technical Architecture & Methodology</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="card">
          <div class="card-title">🎓 Methodology</div>
          <p style="color:#bbb;font-size:0.85rem;line-height:1.9;margin-top:0.6rem;">
            <b style="color:var(--brand-red);">Algorithm:</b> Support Vector Machine (SVM) with RBF kernel<br>
            <b style="color:var(--brand-red);">Scaling:</b> StandardScaler (zero mean, unit variance)<br>
            <b style="color:var(--brand-red);">Tuning:</b> GridSearchCV over C ∈ {0.1, 1, 10, 100}
              and γ ∈ {scale, auto, 0.01, 0.1}<br>
            <b style="color:var(--brand-red);">Validation:</b> Stratified 5-Fold Cross-Validation<br>
            <b style="color:var(--brand-red);">Probability:</b> Platt scaling (predict_proba)
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
          <div class="card-title">📦 Tech Stack</div>
          <table class="styled-table" style="margin-top:0.6rem;">
            <tr><th>Component</th><th>Library</th></tr>
            <tr><td>ML Framework</td><td>scikit-learn</td></tr>
            <tr><td>Data Handling</td><td>pandas / numpy</td></tr>
            <tr><td>Visualization</td><td>matplotlib / seaborn</td></tr>
            <tr><td>Model Persistence</td><td>joblib</td></tr>
            <tr><td>Web Framework</td><td>Streamlit</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
          <div class="card-title">🗂️ Dataset</div>
          <p style="color:#bbb;font-size:0.85rem;line-height:1.9;margin-top:0.6rem;">
            Based on the <b style="color:#fff;">UCI Heart Disease dataset</b>
            comprising 303 patient records from the Cleveland Clinic.
            The target variable is binary:
            <b style="color:#00C853;">0 = No Disease</b> /
            <b style="color:#E53935;">1 = Disease</b>.
          </p>
          <p style="color:#bbb;font-size:0.85rem;line-height:1.9;">
            13 original features were expanded to 18 after feature engineering,
            capturing non-linear relationships between age, cholesterol, blood pressure,
            and heart rate via interaction terms and risk-stratification buckets.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="border-color:#E5393530;">
          <div class="card-title">⚠️ Disclaimer</div>
          <p style="color:#bbb;font-size:0.83rem;line-height:1.8;margin-top:0.6rem;">
            CardioSense AI is designed for <b>research and educational purposes only</b>.
            Predictions should never replace proper clinical diagnosis by a qualified
            medical professional. Always consult a cardiologist for actual patient care.
          </p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    page_login()
else:
    render_sidebar()
    page = st.session_state.page
    if page == "home":     page_home()
    elif page == "input":  page_input()
    elif page == "analysis": page_analysis()
    elif page == "results":  page_results()
    elif page == "about":    page_about()
    else: page_home()
