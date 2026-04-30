"""
Explainable Fake News Detection AI Dashboard
=============================================
Drop-in replacement for app.py — works with the existing
models/model.pkl and models/vectorizer.pkl artifacts.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from predict import predict

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="FakeShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — dark intelligence aesthetic ───────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg-deep:     #080c12;
    --bg-panel:    #0e1420;
    --bg-card:     #131b28;
    --bg-hover:    #1a2436;
    --border:      #1e2d42;
    --accent-cyan: #00d4ff;
    --accent-red:  #ff3c5a;
    --accent-green:#00e676;
    --accent-amber:#ffb300;
    --text-primary:#e8edf5;
    --text-muted:  #6b7f99;
    --text-dim:    #3d5068;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-primary);
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Text area ── */
textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    border-radius: 8px !important;
}
textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* ── Primary button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0099bb 0%, #005577 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s;
}
[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem !important;
    color: var(--accent-cyan) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: var(--bg-card) !important;
    border-radius: 4px !important;
    height: 8px !important;
}
[data-testid="stProgress"] > div > div > div {
    border-radius: 4px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <span style='font-size:2.2rem'>🛡️</span>
        <h2 style='font-family:"Space Mono",monospace; font-size:1.1rem;
                   letter-spacing:0.12em; color:#00d4ff; margin:0.3rem 0 0;'>
            FAKESHIELD AI
        </h2>
        <p style='color:#3d5068; font-size:0.72rem; margin:0;'>
            Explainable News Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <p style='color:#6b7f99; font-size:0.78rem; line-height:1.7;'>
    FakeShield uses <strong style='color:#e8edf5'>TF-IDF vectorization</strong>
    combined with a trained <strong style='color:#e8edf5'>machine-learning
    classifier</strong> to evaluate the credibility of news articles —
    and explains <em>why</em> it reached that verdict.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <p style='font-family:"Space Mono",monospace; font-size:0.7rem;
              letter-spacing:0.1em; color:#00d4ff; margin-bottom:0.5rem;'>
    HOW TO USE
    </p>
    """, unsafe_allow_html=True)

    for step, text in [
        ("01", "Paste or type a news article in the text box"),
        ("02", "Click **Analyse Article**"),
        ("03", "Review the verdict, confidence, and word importance"),
    ]:
        st.markdown(f"""
        <div style='display:flex; gap:0.8rem; margin-bottom:0.6rem;
                    align-items:flex-start;'>
            <span style='font-family:"Space Mono",monospace; color:#00d4ff;
                         font-size:0.7rem; margin-top:2px; min-width:22px;'>
                {step}
            </span>
            <span style='color:#8a9ab5; font-size:0.78rem; line-height:1.6;'>
                {text}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <p style='font-family:"Space Mono",monospace; font-size:0.7rem;
              letter-spacing:0.1em; color:#00d4ff; margin-bottom:0.6rem;'>
    MODEL INFO
    </p>
    """, unsafe_allow_html=True)

    for label, val in [
        ("Vectorizer",  "TF-IDF"),
        ("Framework",   "Scikit-learn"),
        ("Input",       "Raw article text"),
        ("Output",      "FAKE / REAL + probs"),
        ("Explainability", "TF-IDF feature weights"),
    ]:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between;
                    margin-bottom:0.35rem;'>
            <span style='color:#3d5068; font-size:0.72rem;'>{label}</span>
            <span style='color:#8a9ab5; font-size:0.72rem;'>{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style='color:#3d5068; font-size:0.68rem; text-align:center;
              margin-top:0.5rem;'>
    Built with Streamlit · Scikit-learn<br>
    <em>For educational purposes only.</em>
    </p>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding: 2rem 0 1.5rem;'>
    <h1 style='font-family:"Space Mono",monospace; font-size:1.9rem;
               letter-spacing:0.06em; color:#e8edf5; margin:0;'>
        Explainable Fake News
        <span style='color:#00d4ff;'>Detection</span>
    </h1>
    <p style='color:#6b7f99; font-size:0.9rem; margin:0.4rem 0 0;'>
        AI-powered credibility analysis with transparent reasoning
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT PANEL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p style='font-family:"Space Mono",monospace; font-size:0.72rem;
          letter-spacing:0.12em; color:#00d4ff; margin-bottom:0.5rem;'>
◈ ARTICLE INPUT
</p>
""", unsafe_allow_html=True)

article_text = st.text_area(
    label="",
    placeholder="Paste your news article here…",
    height=220,
    label_visibility="collapsed",
)

_, btn_col, _ = st.columns([3, 2, 3])
with btn_col:
    run_analysis = st.button("⚡  ANALYSE ARTICLE", use_container_width=True)

# ── empty-input warning ────────────────────────────────────────────────────────
if run_analysis and not article_text.strip():
    st.markdown("""
    <div style='background:#1a1000; border:1px solid #ffb300;
                border-radius:8px; padding:0.8rem 1.2rem; margin-top:1rem;'>
        ⚠️ <span style='color:#ffb300; font-size:0.88rem;'>
        Please enter some article text before running analysis.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if run_analysis and article_text.strip():

    with st.spinner("Analysing article…"):
        result = predict(article_text)

    label      = result["label"]           # "FAKE" or "REAL"
    confidence = result["confidence"]      # 0-1
    fake_prob  = result["fake_prob"]       # 0-1
    real_prob  = result["real_prob"]       # 0-1
    top_words  = result["top_words"]       # [(word, weight), …]
    word_count = result["word_count"]

    is_fake    = (label == "FAKE")
    verdict_color  = "#ff3c5a" if is_fake else "#00e676"
    verdict_emoji  = "🚨" if is_fake else "✅"
    verdict_bg     = "rgba(255,60,90,0.08)" if is_fake else "rgba(0,230,118,0.08)"
    verdict_border = "#ff3c5a" if is_fake else "#00e676"

    st.markdown("---")

    # ── Verdict banner ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:{verdict_bg}; border:1px solid {verdict_border};
                border-radius:12px; padding:1.4rem 2rem; margin-bottom:1.5rem;
                display:flex; align-items:center; gap:1.5rem;'>
        <span style='font-size:2.8rem; line-height:1;'>{verdict_emoji}</span>
        <div>
            <p style='font-family:"Space Mono",monospace; font-size:0.7rem;
                      letter-spacing:0.15em; color:#6b7f99; margin:0;'>
                VERDICT
            </p>
            <h2 style='font-family:"Space Mono",monospace; font-size:2.2rem;
                       color:{verdict_color}; margin:0.1rem 0 0;'>
                {label} NEWS
            </h2>
            <p style='color:#8a9ab5; font-size:0.82rem; margin:0.3rem 0 0;'>
                Model confidence:&nbsp;
                <strong style='color:{verdict_color};'>
                    {confidence*100:.1f}%
                </strong>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Three-column metrics ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴  FAKE probability", f"{fake_prob*100:.1f}%")
    with col2:
        st.metric("🟢  REAL probability", f"{real_prob*100:.1f}%")
    with col3:
        st.metric("📄  Article word count", f"{word_count:,}")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Probability distribution ──────────────────────────────────────────────
    st.markdown("""
    <p style='font-family:"Space Mono",monospace; font-size:0.72rem;
              letter-spacing:0.12em; color:#00d4ff; margin-bottom:0.4rem;'>
    ◈ PROBABILITY DISTRIBUTION
    </p>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between;
                    margin-bottom:4px;'>
            <span style='color:#ff3c5a; font-size:0.8rem;
                         font-family:"Space Mono",monospace;'>FAKE</span>
            <span style='color:#6b7f99; font-size:0.8rem;'>
                {fake_prob*100:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(fake_prob)

    with col_b:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between;
                    margin-bottom:4px;'>
            <span style='color:#00e676; font-size:0.8rem;
                         font-family:"Space Mono",monospace;'>REAL</span>
            <span style='color:#6b7f99; font-size:0.8rem;'>
                {real_prob*100:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(real_prob)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Top influential words ─────────────────────────────────────────────────
    st.markdown("""
    <p style='font-family:"Space Mono",monospace; font-size:0.72rem;
              letter-spacing:0.12em; color:#00d4ff; margin-bottom:0.8rem;'>
    ◈ MOST INFLUENTIAL WORDS  <span style='color:#3d5068; font-weight:400;
       font-size:0.65rem;'>(by TF-IDF weight)</span>
    </p>
    """, unsafe_allow_html=True)

    if top_words:
        max_w = top_words[0][1] if top_words else 1.0

        # render in two columns of word chips
        half = (len(top_words) + 1) // 2
        wc1, wc2 = st.columns(2)

        for col_words, col_widget in [(top_words[:half], wc1),
                                      (top_words[half:], wc2)]:
            with col_widget:
                for word, weight in col_words:
                    bar_pct = int((weight / max_w) * 100)
                    st.markdown(f"""
                    <div style='background:var(--bg-card,#131b28);
                                border:1px solid #1e2d42; border-radius:6px;
                                padding:0.5rem 0.9rem; margin-bottom:6px;'>
                        <div style='display:flex; justify-content:space-between;
                                    align-items:center; margin-bottom:4px;'>
                            <span style='font-family:"Space Mono",monospace;
                                         color:#e8edf5; font-size:0.82rem;'>
                                {word}
                            </span>
                            <span style='color:#3d5068; font-size:0.7rem;'>
                                {weight:.4f}
                            </span>
                        </div>
                        <div style='background:#0e1420; border-radius:3px;
                                    height:4px; width:100%;'>
                            <div style='background:#00d4ff; border-radius:3px;
                                        height:4px; width:{bar_pct}%;'>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p style='color:#3d5068; font-size:0.82rem;'>
            No significant TF-IDF features found in this text.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Article statistics ────────────────────────────────────────────────────
    st.markdown("""
    <p style='font-family:"Space Mono",monospace; font-size:0.72rem;
              letter-spacing:0.12em; color:#00d4ff; margin-bottom:0.8rem;'>
    ◈ ARTICLE STATISTICS
    </p>
    """, unsafe_allow_html=True)

    char_count = len(article_text)
    sent_count = article_text.count(".") + article_text.count("!") + \
                 article_text.count("?")
    avg_word_len = (
        round(sum(len(w) for w in article_text.split()) / max(word_count, 1), 1)
    )

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Words",      f"{word_count:,}")
    sc2.metric("Characters", f"{char_count:,}")
    sc3.metric("Sentences",  f"{sent_count:,}")
    sc4.metric("Avg word len", f"{avg_word_len}")

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
