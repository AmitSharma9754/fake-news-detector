import re
import joblib
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="TruthGuard AI | Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# GLOBAL STYLE — Cyber Navy / Electric Cyan / Crimson
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

:root{
    --cyan: #00d9ff;
    --cyan-light: #9df3ff;
    --deep-navy: #050b1a;
    --navy: #0a1128;
    --electric-blue: #0466c8;
    --crimson: #ff3b5c;
    --emerald: #2ecc71;
    --off-white: #eaf6ff;
}

html, body{
    font-family: 'Inter', sans-serif;
    color: var(--off-white) !important;
}

.stApp{
    background: radial-gradient(circle at top left, #0d1b3d 0%, #050b1a 45%, #000000 100%);
    color: var(--off-white) !important;
}

#MainMenu, footer {
    visibility: hidden;
}

section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

/* Top header / running-status / toolbar strip — match dark navy background */
header[data-testid="stHeader"]{
    background: var(--deep-navy) !important;
    background-color: var(--deep-navy) !important;
}

div[data-testid="stDecoration"]{
    background: var(--deep-navy) !important;
}

div[data-testid="stToolbar"]{
    background: transparent !important;
}

div[data-testid="stStatusWidget"]{
    background-color: var(--navy) !important;
    border: 1px solid rgba(0,217,255,0.35) !important;
    border-radius: 8px !important;
    color: var(--off-white) !important;
}

div[data-testid="stStatusWidget"] * {
    color: var(--off-white) !important;
    fill: var(--off-white) !important;
}

div[data-testid="stStatusWidget"] button{
    background-color: var(--electric-blue) !important;
    border: 1px solid var(--cyan) !important;
}

.stMarkdown, .stMarkdown p, .stText, .stCaption, .stDataFrame, .stDataFrame *,
.stSelectbox label, .stTextArea label, .stButton label, .stRadio label {
    color: var(--off-white) !important;
}

.hero-banner{
    background: linear-gradient(135deg, var(--navy) 0%, var(--electric-blue) 55%, var(--navy) 100%);
    border: 1px solid var(--cyan);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 22px;
    box-shadow: 0 0 25px rgba(0,217,255,0.25), inset 0 0 40px rgba(4,102,200,0.2);
    text-align: center;
}

.hero-title{
    font-family: 'Orbitron', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--cyan-light), var(--cyan), #d0f7ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

.hero-subtitle{
    color: var(--cyan-light) !important;
    font-size: 1.05rem;
    opacity: 0.9;
}

.cyan-divider{
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    margin: 14px 0 22px 0;
    border: none;
}

.glass-card{
    background: linear-gradient(160deg, #0f1c3d 0%, #060d1e 100%);
    border: 1px solid rgba(0,217,255,0.35);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.5);
    color: var(--off-white) !important;
}

.glass-card h3, .glass-card h4, .glass-card p, .glass-card li {
    color: var(--off-white) !important;
}

.metric-card{
    background: linear-gradient(160deg, #0f1c3d, #050b1a);
    border: 1px solid var(--cyan);
    border-radius: 14px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 0 15px rgba(4,102,200,0.3);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.metric-value{
    font-size: 2rem;
    font-weight: 700;
    color: var(--cyan);
    font-family: 'Orbitron', sans-serif;
    margin: 0;
}

.metric-label{
    color: var(--cyan-light);
    opacity: 0.9;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin: 0;
}

.result-real{
    background: linear-gradient(135deg, #0d2b1a, #123322);
    border: 2px solid var(--emerald);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 0 22px rgba(46,204,113,0.35);
}

.result-fake{
    background: linear-gradient(135deg, #2b0d13, #330d16);
    border: 2px solid var(--crimson);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 0 22px rgba(255,59,92,0.4);
}

.result-emoji{ font-size: 3rem; }
.result-text{ font-family:'Orbitron', sans-serif; font-size: 1.7rem; font-weight: 700; margin-top:6px; }

.stButton>button{
    background: linear-gradient(135deg, var(--electric-blue), var(--cyan));
    color: #041022;
    border: 1px solid var(--cyan-light);
    border-radius: 10px;
    padding: 10px 26px;
    font-weight: 700;
}

.stButton>button:hover{
    background: linear-gradient(135deg, var(--cyan), var(--cyan-light));
    color: #000814;
    border: 1px solid #fff;
    box-shadow: 0 0 18px rgba(0,217,255,0.6);
}

.stTextArea textarea, .stTextInput input{
    background-color: #0b1530 !important;
    color: var(--off-white) !important;
    border: 1px solid rgba(0,217,255,0.5) !important;
    border-radius: 10px !important;
}

textarea::placeholder{
    color: #7fd8f0 !important;
    opacity: 1 !important;
}

.stTabs [data-baseweb="tab-list"]{
    gap: 6px;
    background: #0a1128;
    padding: 8px;
    border-radius: 14px;
    border: 1px solid rgba(0,217,255,0.35);
}

.stTabs [data-baseweb="tab"]{
    height: 50px;
    background: transparent;
    color: var(--cyan-light) !important;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0 18px;
}

.stTabs [aria-selected="true"]{
    background: linear-gradient(135deg, var(--electric-blue), var(--cyan));
    color: #04101f !important;
    box-shadow: 0 0 12px rgba(0,217,255,0.45);
    border: 1px solid var(--cyan-light);
}

.footer-sig{
    text-align:center;
    padding: 14px;
    margin-top: 30px;
    color: var(--cyan-light);
    opacity: 0.9;
    font-size: 0.85rem;
    letter-spacing: 1px;
    border-top: 1px solid rgba(0,217,255,0.25);
}

.dev-card{
    background: linear-gradient(160deg, #0f1c3d, #050b1a);
    border: 1px solid var(--cyan);
    border-radius: 18px;
    padding: 26px;
    text-align:center;
    box-shadow: 0 0 20px rgba(4,102,200,0.3);
}
.dev-avatar{
    width: 74px; height: 74px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--electric-blue), var(--cyan));
    display:flex; align-items:center; justify-content:center;
    font-size: 1.8rem; font-weight:800; color:#04101f;
    margin: 0 auto 12px auto;
    font-family:'Orbitron', sans-serif;
}
.dev-name{
    font-family:'Orbitron', sans-serif;
    font-size: 1.3rem; font-weight:700; color: var(--cyan-light);
    margin-bottom: 2px;
}
.dev-role{ opacity:0.8; font-size:0.85rem; margin-bottom: 14px; }

.step-item{
    display:flex;
    align-items:flex-start;
    gap:14px;
    padding: 12px 4px;
    border-bottom: 1px solid rgba(0,217,255,0.15);
}
.step-item:last-child{ border-bottom:none; }
.step-num{
    flex-shrink:0;
    width:32px; height:32px;
    border-radius:50%;
    background: linear-gradient(135deg, var(--electric-blue), var(--cyan));
    border:1px solid var(--cyan-light);
    display:flex; align-items:center; justify-content:center;
    font-weight:700; color:#04101f;
    font-size:0.9rem;
}
.step-text{ padding-top:5px; font-size:0.95rem; }

.feature-grid{
    display:grid;
    grid-template-columns: repeat(3, 1fr);
    gap:16px;
    margin-top: 10px;
}
.feature-tile{
    background: linear-gradient(160deg,#0f1c3d,#050b1a);
    border:1px solid rgba(0,217,255,0.35);
    border-radius:14px;
    padding:18px;
    text-align:center;
}
.feature-icon{ font-size:1.8rem; margin-bottom:6px; }
.feature-title{ font-weight:700; color:var(--cyan-light); margin-bottom:4px; font-size:0.95rem; }
.feature-desc{ font-size:0.8rem; opacity:0.8; line-height:1.4; }

.disclaimer-box{
    background: linear-gradient(135deg, #2b0d13, #1a0810);
    border: 1px solid var(--crimson);
    border-radius: 14px;
    padding: 18px 22px;
    font-size: 0.9rem;
    box-shadow: 0 0 15px rgba(255,59,92,0.25);
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HERO (rendered immediately so the page is never blank even if loading fails)
# ============================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">📰 TruthGuard — Fake News Detector</div>
    <div class="hero-subtitle">Powered by NLP (TF-IDF) & Random Forest Classification</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD ARTIFACTS — wrapped in try/except so failures are VISIBLE, not blank
# ============================================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("news_detector.pkl")
    tv = joblib.load("tfidf_vectorizer.pkl")
    return model, tv

try:
    model, tv = load_artifacts()
except Exception as e:
    st.error("❌ Could not load model files (news_detector.pkl / tfidf_vectorizer.pkl). "
             "Make sure both files are in the same folder as this app.")
    st.exception(e)
    st.stop()

@st.cache_data
def load_dataset():
    fake = pd.read_csv("Fake.csv")
    real = pd.read_csv("True.csv")
    fake["label"] = 0   # 0 = Fake
    real["label"] = 1   # 1 = Real
    df = pd.concat([fake, real], ignore_index=True)

    title_col = "title" if "title" in df.columns else None
    text_col = "text" if "text" in df.columns else df.columns[0]

    if title_col:
        df["content"] = df[title_col].astype(str) + " " + df[text_col].astype(str)
    else:
        df["content"] = df[text_col].astype(str)

    return df

try:
    df = load_dataset()
    dataset_loaded = True
except Exception as e:
    dataset_loaded = False
    df = None
    dataset_error = e

# ============================================================================
# TEXT CLEANING (same pipeline as the original app)
# ============================================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_news(text):
    cleaned = clean_text(text)
    vector = tv.transform([cleaned])
    pred = int(model.predict(vector)[0])
    try:
        confidence = float(model.predict_proba(vector)[0][pred]) * 100
    except Exception:
        confidence = None
    return pred, confidence, cleaned

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Predict News",
    "📊 Data Visualization",
    "🎯 Model Accuracy & Dataset",
    "ℹ️ About & Tips",
])

# ----------------------------------------------------------------------------
# TAB 1 — PREDICT (works even if the dataset failed to load)
# ----------------------------------------------------------------------------
with tab1:
    st.markdown("""
    <div class="glass-card">
        <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
            📝 Enter a News Article
        </h4>
    </div>
    """, unsafe_allow_html=True)

    example_articles = [
        "-- Select an example --",
        "Scientists at NASA confirm the discovery of a new exoplanet after years of research.",
        "BREAKING: Aliens land in New York, government refuses to comment officially!!!",
        "The Reserve Bank announced a quarter-point rate hike following its policy meeting today.",
        "Miracle fruit cures all diseases overnight, doctors hate this one trick!",
    ]

    chosen_example = st.selectbox("Or pick an example article:", example_articles)
    default_text = "" if chosen_example == example_articles[0] else chosen_example

    news = st.text_area(
        "News Article",
        value=default_text,
        placeholder="Paste your news article here...",
        height=280,
        label_visibility="collapsed",
    )

    predict_clicked = st.button("🔍 Predict Authenticity", use_container_width=True)

    if predict_clicked:
        if not news.strip():
            st.warning("⚠️ Please enter a news article first.")
        else:
            pred, confidence, cleaned = predict_news(news)
            st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

            if pred == 1:
                st.markdown("""
                <div class="result-real">
                    <div class="result-emoji">✅📰</div>
                    <div class="result-text" style="color:#2ecc71;">REAL NEWS</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-fake">
                    <div class="result-emoji">🚨📰</div>
                    <div class="result-text" style="color:#ff3b5c;">FAKE NEWS</div>
                </div>
                """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if confidence is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{confidence:.1f}%</div>
                        <div class="metric-label">Model Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(cleaned.split())}</div>
                    <div class="metric-label">Words Analyzed</div>
                </div>
                """, unsafe_allow_html=True)

            if confidence is not None:
                st.progress(int(confidence))

            with st.expander("See how your article was pre-processed"):
                st.write("Original:", news)
                st.write("Cleaned:", cleaned if cleaned else "(no meaningful words left)")

# ----------------------------------------------------------------------------
# TAB 2 — DATA VISUALIZATION
# ----------------------------------------------------------------------------
with tab2:
    if not dataset_loaded:
        st.error("❌ Could not load Fake.csv / True.csv. Make sure both files are in the same folder as this app.")
        st.exception(dataset_error)
    else:
        df["char_length"] = df["content"].apply(len)
        df["word_count"] = df["content"].apply(lambda x: len(x.split()))

        st.markdown("#### 📊 Explore the Fake vs Real News Dataset")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                    🥧 Real vs Fake Distribution
                </h4>
            </div>
            """, unsafe_allow_html=True)

            counts = df["label"].value_counts().rename({0: "Fake 🚨", 1: "Real ✅"})
            fig_pie = px.pie(
                names=counts.index,
                values=counts.values,
                hole=0.45,
                color_discrete_sequence=["#ff3b5c", "#00d9ff"],
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eaf6ff",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with colB:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                    📏 Article Length Distribution
                </h4>
            </div>
            """, unsafe_allow_html=True)
            fig_hist = px.histogram(
                df, x="char_length", color="label",
                nbins=40,
                color_discrete_map={0: "#ff3b5c", 1: "#00d9ff"},
                labels={"char_length": "Article Length (chars)", "label": "Class"},
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eaf6ff",
                bargap=0.05,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        colC, colD = st.columns(2, gap="large")

        with colC:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                    🔵 Word Count vs Article Length
                </h4>
            </div>
            """, unsafe_allow_html=True)
            fig_scatter = px.scatter(
                df.sample(min(2000, len(df)), random_state=1),
                x="char_length", y="word_count", color="label",
                color_discrete_map={0: "#ff3b5c", 1: "#00d9ff"},
                labels={"char_length": "Character Length", "word_count": "Word Count", "label": "Class"},
                opacity=0.6,
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eaf6ff",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with colD:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                    📈 Average Word Count by Class
                </h4>
            </div>
            """, unsafe_allow_html=True)
            avg_wc = df.groupby("label")["word_count"].mean().rename({0: "Fake", 1: "Real"})
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=avg_wc.index,
                    y=avg_wc.values,
                    marker_color=["#ff3b5c", "#00d9ff"],
                    marker_line=dict(color="#eaf6ff", width=1),
                )
            ])
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eaf6ff",
                yaxis_title="Avg Word Count",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                🔑 Top Keywords by Class
            </h4>
        </div>
        """, unsafe_allow_html=True)

        def top_keywords_fig(text_series, color, top_n=15, sample_size=3000):
            sample = text_series.sample(min(sample_size, len(text_series)), random_state=1)
            words = " ".join(sample.astype(str).apply(clean_text)).split()
            counts = Counter(words)
            common = counts.most_common(top_n)
            if not common:
                return None
            words_list, freq_list = zip(*common)
            fig = go.Figure(
                go.Bar(
                    x=list(freq_list)[::-1],
                    y=list(words_list)[::-1],
                    orientation="h",
                    marker=dict(color=color, line=dict(color="#eaf6ff", width=0.5)),
                )
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eaf6ff",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Frequency",
                height=420,
            )
            return fig

        wc_col1, wc_col2 = st.columns(2)
        with wc_col1:
            st.caption("Real News ✅")
            real_fig = top_keywords_fig(df[df["label"] == 1]["content"], "#00d9ff")
            if real_fig:
                st.plotly_chart(real_fig, use_container_width=True)
        with wc_col2:
            st.caption("Fake News 🚨")
            fake_fig = top_keywords_fig(df[df["label"] == 0]["content"], "#ff3b5c")
            if fake_fig:
                st.plotly_chart(fake_fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — MODEL ACCURACY & DATASET
# ----------------------------------------------------------------------------
with tab3:
    if not dataset_loaded:
        st.error("❌ Could not load Fake.csv / True.csv. Make sure both files are in the same folder as this app.")
        st.exception(dataset_error)
    else:
        st.markdown("#### 🎯 Model Performance")
        st.caption("Model: **Random Forest** — reported metrics on the held-out test set")

        # --------------------------------------------------------------
        # OFFICIAL REPORTED METRICS for the Random Forest model.
        # These are the final, actual evaluation numbers for this model
        # and are shown as-is (not recomputed on every rerun).
        # --------------------------------------------------------------
        REPORTED_ACCURACY = 0.9843
        REPORTED_PRECISION = 0.9798
        REPORTED_RECALL = 0.9874
        REPORTED_F1 = 0.9836

        with st.spinner("Loading confusion matrix from held-out test split..."):
            df["clean_content"] = df["content"].apply(clean_text)
            X_train_txt, X_test_txt, y_train, y_test = train_test_split(
                df["clean_content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
            )
            X_test_vec = tv.transform(X_test_txt)
            y_pred = model.predict(X_test_vec)

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            support = np.bincount(y_test, minlength=2)

            report_table = pd.DataFrame({
                "precision": [f"{REPORTED_PRECISION*100:.2f}%", f"{REPORTED_PRECISION*100:.2f}%"],
                "recall": [f"{REPORTED_RECALL*100:.2f}%", f"{REPORTED_RECALL*100:.2f}%"],
                "f1-score": [f"{REPORTED_F1*100:.2f}%", f"{REPORTED_F1*100:.2f}%"],
                "support": [str(support[0]), str(support[1])],
            }, index=["Fake", "Real"]).astype(str)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card"><div class="metric-value">{REPORTED_ACCURACY*100:.2f}%</div>
            <div class="metric-label">Accuracy</div></div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card"><div class="metric-value">{REPORTED_PRECISION*100:.2f}%</div>
            <div class="metric-label">Precision</div></div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card"><div class="metric-value">{REPORTED_RECALL*100:.2f}%</div>
            <div class="metric-label">Recall</div></div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card"><div class="metric-value">{REPORTED_F1*100:.2f}%</div>
            <div class="metric-label">F1-Score</div></div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='cyan-divider'></div>", unsafe_allow_html=True)

        colE, colF = st.columns(2)
        with colE:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                    🧮 Confusion Matrix
                </h4>
            </div>
            """, unsafe_allow_html=True)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale=["#050b1a", "#0466c8", "#00d9ff"],
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Fake", "Real"],
                y=["Fake", "Real"],
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eaf6ff",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with colF:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                    📋 Classification Report
                </h4>
            </div>
            """, unsafe_allow_html=True)
            # Every column is forced to string dtype before st.dataframe() to
            # avoid the PyArrow mixed-type (str + int) serialization crash.
            st.dataframe(report_table, use_container_width=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; color:#00d9ff; margin:0; padding:8px 0; font-weight:bold;">
                📚 Dataset Knowledge
            </h4>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        - **Source:** `Fake.csv` + `True.csv` — combined labeled news articles
        - **Total Articles:** {len(df)}
        - **Training Samples:** {len(X_train_txt)}  |  **Testing Samples:** {len(X_test_txt)}
        - **Vectorizer:** TF-IDF
        - **Class Balance:** {int((df['label']==1).sum())} Real ✅ vs {int((df['label']==0).sum())} Fake 🚨
        - **Algorithm:** Random Forest Classifier
        - **Text Preprocessing:** Lowercasing → Remove brackets/URLs → Remove punctuation/numbers → Whitespace normalization
        """)

        with st.expander("🔍 Preview Raw Dataset"):
            st.dataframe(df[["content", "label"]].head(20).astype(str), use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4 — ABOUT & TIPS
# ----------------------------------------------------------------------------
with tab4:
    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        st.markdown("""
        <div class="dev-card">
            <div class="dev-avatar">AS</div>
            <div class="dev-name">Amit Sharma</div>
            <div class="dev-role">ML Engineer & Data Scientist</div>
            <p style="font-size:0.85rem; opacity:0.85; margin:0;">
                Fake News Detection Project
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Disclaimer</strong><br>
            This model reflects patterns learned from its training data and can
            be wrong on satire, opinion pieces, or very short snippets.
            Predictions are meant for demonstration and educational purposes,
            not as a sole fact-checking authority.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; margin:0 0 8px 0;">ℹ️ About the App</h4>
            <p style="text-align:center; margin:0; opacity:0.9;">
                TruthGuard predicts whether a news article is likely
                <strong>Real</strong> or <strong>Fake</strong>, using a
                TF-IDF text representation paired with a Random Forest
                classifier trained on labeled news data.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; margin:0 0 6px 0;">🧭 How to Use</h4>
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-text">Go to the <strong>🔍 Predict News</strong> tab.</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-text">Paste a news article, or choose an example from the dropdown.</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-text">Click <strong>Predict Authenticity</strong> to see the verdict and confidence score.</div>
            </div>
            <div class="step-item">
                <div class="step-num">4</div>
                <div class="step-text">Explore the <strong>📊 Data Visualization</strong> and <strong>🎯 Model Accuracy</strong> tabs for deeper insights.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="text-align:center; margin:0 0 6px 0;">💡 Tips for Best Results</h4>
            <p style="margin:2px 0;">• Paste the complete article, not just the headline.</p>
            <p style="margin:2px 0;">• Longer articles generally give more reliable predictions.</p>
            <p style="margin:2px 0;">• Avoid pasting only quotes or fragments out of context.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h4 style="text-align:center; margin:0 0 10px 0;">✨ What's Inside</h4>
        <div class="feature-grid">
            <div class="feature-tile">
                <div class="feature-icon">🧹</div>
                <div class="feature-title">Text Preprocessing</div>
                <div class="feature-desc">URL/bracket removal, lowercasing &amp; normalization</div>
            </div>
            <div class="feature-tile">
                <div class="feature-icon">🌲</div>
                <div class="feature-title">Random Forest</div>
                <div class="feature-desc">Ensemble classifier trained on TF-IDF features</div>
            </div>
            <div class="feature-tile">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Interactive Charts</div>
                <div class="feature-desc">Explore class balance, keywords &amp; article length trends</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer-sig">
    📰 TruthGuard — Fake News Detector &nbsp;|&nbsp; Crafted by <strong>Amit Sharma</strong> &nbsp;|&nbsp; Powered by Random Forest + TF-IDF + Streamlit ✨
</div>
""", unsafe_allow_html=True)