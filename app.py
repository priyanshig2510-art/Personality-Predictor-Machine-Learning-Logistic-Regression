"""
Personality Predictor – Streamlit UI
Uses personality_model.pkl + scaler.pkl exported from logistic_solutions.ipynb

Run:  streamlit run app.py
"""

import pickle
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent

# LabelEncoder sorts labels alphabetically -> 0 = Ambivert, 1 = Extrovert, 2 = Introvert
CLASS_NAMES = ["Ambivert", "Extrovert", "Introvert"]

PERSONA = {
    "Ambivert": {
        "emoji": "⚖️",
        "color": "#8B5CF6",
        "tagline": "The balanced one",
        "desc": "You flex between social and solo modes with ease. You can enjoy a lively "
        "gathering and a quiet evening alone equally well.",
    },
    "Extrovert": {
        "emoji": "🎉",
        "color": "#F59E0B",
        "tagline": "The energizer",
        "desc": "You recharge around people. Conversations, group activities and new "
        "experiences give you energy and keep you going.",
    },
    "Introvert": {
        "emoji": "🌙",
        "color": "#3B82F6",
        "tagline": "The deep thinker",
        "desc": "You recharge in calm, quiet settings. You prefer meaningful one-to-one "
        "conversations and need time alone to reflect.",
    },
}

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
    #MainMenu, footer { visibility: hidden; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 55%, #F093FB 100%);
        border-radius: 24px;
        padding: 2.4rem 2.6rem;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 18px 40px rgba(118, 75, 162, 0.35);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "🧠";
        position: absolute; right: 2rem; top: 50%;
        transform: translateY(-50%);
        font-size: 7rem; opacity: .18;
    }
    .hero h1 { margin: 0; font-size: 2.4rem; font-weight: 700; color: white; }
    .hero p  { margin: .5rem 0 0; font-size: 1.05rem; opacity: .92; max-width: 640px; }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(6px);
    }
    .card h3 { margin-top: 0; font-weight: 600; }

    /* Result */
    .result {
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 14px 34px rgba(0,0,0,.25);
        animation: pop .45s ease;
    }
    @keyframes pop { from { transform: scale(.94); opacity: 0; } to { transform: scale(1); opacity: 1; } }
    .result .emoji { font-size: 4.2rem; line-height: 1; }
    .result .label { font-size: 2.2rem; font-weight: 700; margin: .4rem 0 0; }
    .result .tag   { font-size: 1rem; opacity: .9; letter-spacing: .04em; text-transform: uppercase; }
    .result .conf  { margin-top: .9rem; font-size: 1.1rem; font-weight: 500;
                     background: rgba(255,255,255,.22); display: inline-block;
                     padding: .35rem 1rem; border-radius: 999px; }
    .result .desc  { margin-top: 1rem; font-size: .98rem; opacity: .95; }

    /* Metric tiles */
    .tile {
        border-radius: 16px; padding: 1rem 1.2rem;
        background: linear-gradient(135deg, rgba(102,126,234,.18), rgba(118,75,162,.18));
        border: 1px solid rgba(128,128,128,.2);
    }
    .tile .k { font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em; }
    .tile .v { font-size: 1.7rem; font-weight: 700; }

    /* Button */
    div.stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #667EEA, #764BA2);
        color: white; border: none; border-radius: 14px;
        padding: .75rem 1.4rem; font-weight: 600; font-size: 1rem;
        transition: transform .15s ease, box-shadow .15s ease;
        width: 100%;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(118,75,162,.45);
        color: white; border: none;
    }

    /* Tabs */
    button[data-baseweb="tab"] { font-weight: 600; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Load model + scaler
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pickle_from_path(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_artifacts():
    model_path = BASE_DIR / "personality_model.pkl"
    scaler_path = BASE_DIR / "scaler.pkl"

    if model_path.exists() and scaler_path.exists():
        return load_pickle_from_path(str(model_path)), load_pickle_from_path(str(scaler_path))

    st.sidebar.warning("Model files not found next to app.py. Upload them below.")
    m_file = st.sidebar.file_uploader("personality_model.pkl", type="pkl")
    s_file = st.sidebar.file_uploader("scaler.pkl", type="pkl")
    if m_file and s_file:
        return pickle.load(m_file), pickle.load(s_file)
    return None, None


model, scaler = get_artifacts()

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Personality Predictor</h1>
        <p>Adjust the behavioural and social traits below and let a Logistic Regression
        model tell you whether the profile looks <b>Introvert</b>, <b>Ambivert</b> or
        <b>Extrovert</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None or scaler is None:
    st.info(
        "👈 Place **personality_model.pkl** and **scaler.pkl** (exported from your notebook) "
        "in the same folder as `app.py`, or upload them from the sidebar."
    )
    st.stop()

# Feature names come straight from the fitted scaler (fit on a DataFrame in the notebook)
if hasattr(scaler, "feature_names_in_"):
    features = list(scaler.feature_names_in_)
else:
    features = [f"feature_{i + 1}" for i in range(scaler.n_features_in_)]

means = scaler.mean_
stds = scaler.scale_


def slider_range(mean: float, std: float):
    lo, hi = mean - 3 * std, mean + 3 * std
    if mean > 1.5 * std:  # data looks strictly positive (scores, ratings...)
        lo = max(lo, 0.0)
    return float(np.floor(lo)), float(np.ceil(hi))


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.caption("Model: Logistic Regression")
    st.markdown("---")
    st.markdown("**Features used by the model**")
    for f in features:
        st.markdown(f"• {pretty(f)}")
    st.markdown("---")
    if st.button("🔄 Reset sliders to average"):
        for f, m in zip(features, means):
            st.session_state[f"in_{f}"] = float(m)
        st.rerun()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_predict, tab_insights, tab_data = st.tabs(["🔮 Predict", "📊 Model Insights", "📁 Explore Data"])

# ============================== PREDICT TAB ==================================
with tab_predict:
    left, right = st.columns([1.15, 1], gap="large")

    values = {}
    with left:
        st.markdown('<div class="card"><h3>🎛️ Your Traits</h3>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (f, m, s) in enumerate(zip(features, means, stds)):
            lo, hi = slider_range(m, s)
            key = f"in_{f}"
            if key not in st.session_state:
                st.session_state[key] = float(np.clip(round(m, 1), lo, hi))
            with cols[i % 2]:
                values[f] = st.slider(pretty(f), lo, hi, step=0.1, key=key)
        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction
    X_in = pd.DataFrame([values])[features]
    X_scaled = scaler.transform(X_in)
    proba = model.predict_proba(X_scaled)[0]
    classes = [CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else str(c) for c in model.classes_]
    top_idx = int(np.argmax(proba))
    top = classes[top_idx]
    info = PERSONA.get(top, {"emoji": "🧩", "color": "#667EEA", "tagline": "", "desc": ""})

    with right:
        st.markdown(
            f"""
            <div class="result" style="background: linear-gradient(135deg, {info['color']}, {info['color']}aa);">
                <div class="emoji">{info['emoji']}</div>
                <div class="tag">{info['tagline']}</div>
                <div class="label">{top}</div>
                <div class="conf">Confidence: {proba[top_idx] * 100:.1f}%</div>
                <div class="desc">{info['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Probability breakdown")
        prob_df = pd.DataFrame({"Personality": classes, "Probability": proba})
        color_scale = alt.Scale(
            domain=list(PERSONA.keys()),
            range=[PERSONA[k]["color"] for k in PERSONA],
        )
        chart = (
            alt.Chart(prob_df)
            .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, height=26)
            .encode(
                x=alt.X("Probability:Q", axis=alt.Axis(format="%", title=None), scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Personality:N", sort="-x", title=None),
                color=alt.Color("Personality:N", scale=color_scale, legend=None),
                tooltip=["Personality", alt.Tooltip("Probability:Q", format=".1%")],
            )
            .properties(height=150)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, use_container_width=True)

# ============================ INSIGHTS TAB ===================================
with tab_insights:
    st.markdown("### How each trait pushes the prediction")
    st.caption(
        "Coefficients of the logistic regression (on standardised features). "
        "Larger bars mean the trait has a stronger influence on that personality type."
    )

    coef = model.coef_
    coef_df = pd.DataFrame(coef.T, index=features, columns=classes)
    long_df = coef_df.reset_index().melt("index", var_name="Personality", value_name="Coefficient")
    long_df["Feature"] = long_df["index"].map(pretty)

    heat = (
        alt.Chart(long_df)
        .mark_rect(cornerRadius=6)
        .encode(
            x=alt.X("Personality:N", title=None),
            y=alt.Y("Feature:N", title=None, sort=None),
            color=alt.Color("Coefficient:Q", scale=alt.Scale(scheme="redblue", domainMid=0), title="Weight"),
            tooltip=["Feature", "Personality", alt.Tooltip("Coefficient:Q", format=".3f")],
        )
        .properties(height=max(220, 38 * len(features)))
    )
    text = heat.mark_text(fontSize=12).encode(
        text=alt.Text("Coefficient:Q", format=".2f"),
        color=alt.value("black"),
    )
    st.altair_chart(heat + text, use_container_width=True)

    importance = (
        pd.DataFrame({"Feature": [pretty(f) for f in features], "Importance": np.abs(coef).mean(axis=0)})
        .sort_values("Importance", ascending=False)
    )
    st.markdown("### Overall feature importance")
    imp_chart = (
        alt.Chart(importance)
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, color="#764BA2")
        .encode(
            x=alt.X("Importance:Q", title="Mean |coefficient|"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")],
        )
        .properties(height=max(200, 34 * len(features)))
    )
    st.altair_chart(imp_chart, use_container_width=True)

# ============================== DATA TAB =====================================
with tab_data:
    st.markdown("### Explore your dataset")
    st.caption("Upload the CSV used in the notebook (e.g. *personality_synthetic_dataset.csv*) to browse it.")
    csv = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")

    if csv is not None:
        df = pd.read_csv(csv)

        c1, c2, c3, c4 = st.columns(4)
        tiles = [
            ("Rows", f"{len(df):,}"),
            ("Columns", df.shape[1]),
            ("Missing values", int(df.isnull().sum().sum())),
            ("Duplicates", int(df.duplicated().sum())),
        ]
        for col, (k, v) in zip((c1, c2, c3, c4), tiles):
            col.markdown(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True, height=280)

        if "personality_type" in df.columns:
            st.markdown("#### Personality distribution")
            counts = df["personality_type"].value_counts().reset_index()
            counts.columns = ["Personality", "Count"]
            pie = (
                alt.Chart(counts)
                .mark_arc(innerRadius=60, outerRadius=110)
                .encode(
                    theta="Count:Q",
                    color=alt.Color(
                        "Personality:N",
                        scale=alt.Scale(
                            domain=list(PERSONA.keys()),
                            range=[PERSONA[k]["color"] for k in PERSONA],
                        ),
                    ),
                    tooltip=["Personality", "Count"],
                )
                .properties(height=260)
            )
            st.altair_chart(pie, use_container_width=True)

        num_cols = df.select_dtypes("number").columns.tolist()
        if num_cols:
            st.markdown("#### Feature distribution")
            pick = st.selectbox("Choose a feature", num_cols, format_func=pretty)
            hist = (
                alt.Chart(df)
                .mark_bar(opacity=0.85, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X(f"{pick}:Q", bin=alt.Bin(maxbins=30), title=pretty(pick)),
                    y=alt.Y("count()", title="People"),
                    color=alt.Color(
                        "personality_type:N",
                        title="Personality",
                        scale=alt.Scale(
                            domain=list(PERSONA.keys()),
                            range=[PERSONA[k]["color"] for k in PERSONA],
                        ),
                    )
                    if df["personality_type"].dtype == object
                    else alt.value("#667EEA"),
                )
                .properties(height=300)
            ) if "personality_type" in df.columns else (
                alt.Chart(df)
                .mark_bar(color="#667EEA")
                .encode(x=alt.X(f"{pick}:Q", bin=alt.Bin(maxbins=30)), y="count()")
                .properties(height=300)
            )
            st.altair_chart(hist, use_container_width=True)

            st.markdown("#### Correlation heatmap")
            corr = df[num_cols].corr().reset_index().melt("index")
            corr.columns = ["A", "B", "Correlation"]
            ch = (
                alt.Chart(corr)
                .mark_rect()
                .encode(
                    x=alt.X("A:N", title=None),
                    y=alt.Y("B:N", title=None),
                    color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="purpleorange", domain=[-1, 1])),
                    tooltip=["A", "B", alt.Tooltip("Correlation:Q", format=".2f")],
                )
                .properties(height=380)
            )
            st.altair_chart(ch, use_container_width=True)
    else:
        st.info("No dataset uploaded yet – this tab is optional.")

st.markdown(
    "<p style='text-align:center; opacity:.55; margin-top:2rem;'>Built with Streamlit · Logistic Regression · scikit-learn</p>",
    unsafe_allow_html=True,
)