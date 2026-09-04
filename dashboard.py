"""
dashboard.py — Diabetic Retinopathy Detection | Streamlit Dashboard
Run with: streamlit run dashboard.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# ── Page config (must be first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="DR Detection | EfficientNetB3",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import utils ─────────────────────────────────────────────────────────
from utils import (
    load_model,
    predict,
    get_gradcam_overlay,
    get_sample_images,
    CLASS_LABELS,
    CLASS_NAMES,
    SEVERITY_MAP,
    DATASET_COUNTS,
    CLASSIFICATION_REPORT,
    CONFUSION_MATRIX,
)

# ─────────────────────────────────────────────
# Global CSS — Premium dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 50%, #0a1628 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b35 0%, #0a1525 100%);
    border-right: 1px solid rgba(56, 189, 248, 0.15);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Headers ── */
h1 { 
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
}
h2, h3 { color: #e2e8f0 !important; font-weight: 700 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(56,189,248,0.15);
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.85rem !important; }
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 800 !important; }

/* ── Stat card div ── */
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    margin-bottom: 12px;
}
.stat-card:hover {
    border-color: rgba(56,189,248,0.5);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(56,189,248,0.15);
}
.stat-card h2 { font-size: 2.2rem !important; margin: 0; }
.stat-card p { color: #94a3b8; margin: 4px 0 0 0; font-size: 0.9rem; }

/* ── Severity badge ── */
.severity-badge {
    display: inline-block;
    border-radius: 50px;
    padding: 10px 28px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(56,189,248,0.3) !important;
    border-radius: 16px !important;
    background: rgba(56,189,248,0.03) !important;
    padding: 12px !important;
}

/* ── Section divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent);
    margin: 32px 0;
}

/* ── Info box ── */
.info-box {
    background: rgba(56,189,248,0.07);
    border-left: 4px solid #38bdf8;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 16px 0;
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ── Class pill ── */
.class-pill {
    display: inline-block;
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    color: white !important;
}

/* ── Stremlit default overrides ── */
.stButton>button {
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 28px;
    transition: all 0.3s;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(56,189,248,0.35);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:3rem;'>👁️</div>
        <div style='font-size:1.1rem; font-weight:700; color:#38bdf8; margin-top:6px;'>DR Detection</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:2px;'>EfficientNetB3 · TensorFlow</div>
    </div>
    <hr style='border-color:rgba(56,189,248,0.15); margin:16px 0;'/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "🏠 Home",
            "🔬 Live Prediction",
            "📊 Dataset Analysis",
            "📈 Model Performance",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style='border-color:rgba(56,189,248,0.15); margin:16px 0;'/>
    <div style='font-size:0.75rem; color:#475569; text-align:center; padding-bottom:16px;'>
        <div>Model: <b style='color:#38bdf8;'>EfficientNetB3</b></div>
        <div>Test Accuracy: <b style='color:#22c55e;'>77.38%</b></div>
        <div style='margin-top:8px;'>Classes: 5 DR Grades</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 1 — Home
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("# 👁️ Diabetic Retinopathy Detection")
    st.markdown(
        "<p style='color:#94a3b8; font-size:1.1rem; margin-top:-12px;'>Deep Learning-based severity grading from retinal fundus images</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Key metrics row ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🖼️ Total Images", "3,662", help="Total images in dataset")
    c2.metric("🏷️ Classes", "5", help="DR severity grades")
    c3.metric("🎯 Test Accuracy", "77.38%", help="EfficientNetB3 on held-out test set")
    c4.metric("🧠 Model", "EfficientNetB3", help="Transfer learning from ImageNet")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── What is DR? ──
    st.markdown("## 🩺 What is Diabetic Retinopathy?")
    col_text, col_scale = st.columns([1.4, 1])

    with col_text:
        st.markdown("""
        <div class='info-box'>
        <b>Diabetic Retinopathy (DR)</b> is a diabetes complication that affects eyes. It's caused by damage to the blood vessels 
        of the light-sensitive tissue at the back of the eye (retina). DR is the <b>leading cause of blindness</b> in working-age adults globally.<br><br>
        Early detection through <b>automated fundus image analysis</b> can prevent vision loss in ~90% of cases. 
        This project uses a fine-tuned <b>EfficientNetB3</b> convolutional neural network to classify retinal images 
        into 5 severity grades with high accuracy.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔬 Project Pipeline")
        steps = [
            ("1️⃣", "Data Collection", "3,662 labelled retinal fundus images across 5 DR grades"),
            ("2️⃣", "Preprocessing", "CLAHE enhancement, resizing to 224×224, augmentation"),
            ("3️⃣", "Model Training", "EfficientNetB3 fine-tuned with class-weighted cross-entropy"),
            ("4️⃣", "Evaluation", "77.38% accuracy · Confusion matrix · Classification report"),
            ("5️⃣", "Explainability", "Grad-CAM heatmaps highlight decision regions"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex; align-items:flex-start; gap:14px; margin-bottom:12px;'>
                <span style='font-size:1.4rem; line-height:1.6;'>{icon}</span>
                <div>
                    <div style='font-weight:600; color:#e2e8f0;'>{title}</div>
                    <div style='font-size:0.85rem; color:#94a3b8;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_scale:
        st.markdown("### 🚦 DR Severity Scale")
        for cls, info in SEVERITY_MAP.items():
            st.markdown(f"""
            <div style='
                background: rgba(255,255,255,0.04);
                border-left: 4px solid {info["color"]};
                border-radius: 0 12px 12px 0;
                padding: 12px 16px;
                margin-bottom: 10px;
            '>
                <div style='font-weight:700; color:{info["color"]}; font-size:0.95rem;'>
                    {info["icon"]} {cls.replace("_", " ")}
                </div>
                <div style='font-size:0.8rem; color:#94a3b8; margin-top:2px;'>{info["desc"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Dataset distribution teaser ──
    st.markdown("## 📊 Dataset at a Glance")
    labels = list(DATASET_COUNTS.keys())
    values = list(DATASET_COUNTS.values())
    colors = ["#84cc16", "#f59e0b", "#22c55e", "#7c3aed", "#ef4444"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=values,
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(showgrid=False, tickfont=dict(size=13)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Image Count"),
        margin=dict(t=20, b=10, l=0, r=0),
        height=320,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style='text-align:center; color:#475569; font-size:0.85rem; margin-top:-10px;'>
        Note: The dataset is imbalanced — class weights were applied during training to compensate.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2 — Live Prediction
# ─────────────────────────────────────────────
elif page == "🔬 Live Prediction":
    st.markdown("# 🔬 Live Retinal Image Prediction")
    st.markdown(
        "<p style='color:#94a3b8; font-size:1rem; margin-top:-12px;'>Upload a fundus image to get an instant DR grade + Grad-CAM explainability heatmap</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Load model
    with st.spinner("⚙️ Loading EfficientNetB3 model..."):
        model = load_model()

    if model is None:
        st.error("❌ Could not load the model. Make sure `best_efficientnet_model.keras` is in the project directory.")
        st.stop()

    st.success("✅ Model loaded successfully!")

    # Upload
    uploaded_file = st.file_uploader(
        "Drop a retinal fundus image here",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG",
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Run prediction
        with st.spinner("🧠 Analysing retinal image..."):
            pred_class, confidence, all_probs = predict(model, pil_image)
            gradcam_img = get_gradcam_overlay(model, pil_image)

        info = SEVERITY_MAP[pred_class]

        # ── Result header ──
        st.markdown(f"""
        <div style='text-align:center; margin: 20px 0 28px 0;'>
            <div style='font-size:0.95rem; color:#94a3b8; margin-bottom:8px;'>Prediction Result</div>
            <div class='severity-badge' style='background:{info["color"]}22; border:2px solid {info["color"]}; color:{info["color"]};'>
                {info["icon"]} {pred_class.replace("_", " ")} &nbsp;·&nbsp; {confidence*100:.1f}% confidence
            </div>
            <div style='margin-top:12px; color:#cbd5e1; font-size:0.9rem;'>{info["desc"]}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Image columns ──
        col_orig, col_cam = st.columns(2)

        with col_orig:
            st.markdown("#### 📷 Original Image")
            st.image(pil_image, use_container_width=True, clamp=True)

        with col_cam:
            st.markdown("#### 🔥 Grad-CAM Heatmap")
            if gradcam_img is not None:
                st.image(gradcam_img, use_container_width=True, clamp=True)
                st.markdown("""
                <div style='font-size:0.8rem; color:#64748b; text-align:center; margin-top:4px;'>
                    🔴 Red regions = areas the model focused on for its decision
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Grad-CAM could not be generated for this image.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Confidence bar chart ──
        st.markdown("#### 📊 Class Confidence Scores")
        class_display = [c.replace("_", " ") for c in CLASS_LABELS.values()]
        bar_colors = [SEVERITY_MAP[c]["color"] for c in CLASS_LABELS.values()]

        fig_probs = go.Figure(go.Bar(
            x=class_display,
            y=[float(p) * 100 for p in all_probs],
            marker=dict(color=bar_colors, opacity=0.85),
            text=[f"{p*100:.1f}%" for p in all_probs],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=12),
        ))
        fig_probs.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            xaxis=dict(showgrid=False, tickfont=dict(size=13)),
            yaxis=dict(
                showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                title="Confidence (%)", range=[0, 110],
            ),
            margin=dict(t=20, b=10, l=0, r=0),
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig_probs, use_container_width=True)

        # ── Recommendation ──
        st.markdown("#### 💡 Clinical Guidance")
        guidance = {
            "No_DR":          "No immediate action required. Continue regular annual eye exams.",
            "Mild":           "Early signs detected. Recommend follow-up in 12 months with an ophthalmologist.",
            "Moderate":       "Moderate retinopathy present. Referral to ophthalmologist recommended within 6 months.",
            "Severe":         "Severe NPDR detected. Urgent ophthalmology referral within 1 month is recommended.",
            "Proliferate_DR": "Proliferative DR detected. Immediate ophthalmology referral is required. Risk of vision loss is high.",
        }
        st.markdown(f"""
        <div class='info-box' style='border-left-color:{info["color"]};'>
            {info["icon"]} <b>{pred_class.replace("_"," ")}</b>: {guidance[pred_class]}
        </div>
        <div style='font-size:0.75rem; color:#475569; margin-top:-8px;'>
            ⚠️ This tool is for research purposes only and does not constitute medical advice.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Placeholder when no image uploaded
        st.markdown("""
        <div style='
            text-align:center;
            padding: 60px 20px;
            background: rgba(56,189,248,0.03);
            border: 2px dashed rgba(56,189,248,0.2);
            border-radius: 20px;
            margin: 20px 0;
        '>
            <div style='font-size:4rem; margin-bottom:16px;'>🏥</div>
            <div style='font-size:1.3rem; font-weight:600; color:#e2e8f0; margin-bottom:8px;'>Upload a Retinal Fundus Image</div>
            <div style='color:#64748b; font-size:0.95rem;'>
                Supported: PNG, JPG, JPEG<br>
                The model will classify the image and show a Grad-CAM heatmap
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show sample images from dataset
        st.markdown("#### 📁 Sample Images from Dataset")
        sample_cols = st.columns(5)
        for i, (cls, info) in enumerate(SEVERITY_MAP.items()):
            with sample_cols[i]:
                st.markdown(f"<div style='text-align:center; color:{info['color']}; font-weight:600; font-size:0.8rem; margin-bottom:6px;'>{info['icon']} {cls.replace('_', ' ')}</div>", unsafe_allow_html=True)
                samples = get_sample_images(cls, n=1)
                if samples:
                    st.image(samples[0], use_container_width=True)
                else:
                    st.markdown("<div style='background:rgba(255,255,255,0.04); border-radius:8px; height:100px; display:flex; align-items:center; justify-content:center; color:#475569;'>No sample</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 3 — Dataset Analysis
# ─────────────────────────────────────────────
elif page == "📊 Dataset Analysis":
    st.markdown("# 📊 Dataset Analysis")
    st.markdown(
        "<p style='color:#94a3b8; font-size:1rem; margin-top:-12px;'>Explore class distributions and representative images from each DR grade</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    total = sum(DATASET_COUNTS.values())

    # ── Summary metrics ──
    cols = st.columns(5)
    class_colors = {"No_DR": "#22c55e", "Mild": "#84cc16", "Moderate": "#f59e0b", "Severe": "#ef4444", "Proliferate_DR": "#7c3aed"}
    for i, (cls, cnt) in enumerate(DATASET_COUNTS.items()):
        with cols[i]:
            pct = cnt / total * 100
            st.markdown(f"""
            <div class='stat-card'>
                <div style='font-size:0.8rem; color:{class_colors[cls]}; font-weight:700; margin-bottom:6px;'>
                    {SEVERITY_MAP[cls]["icon"]} {cls.replace("_"," ")}
                </div>
                <h2 style='color:{class_colors[cls]} !important;'>{cnt:,}</h2>
                <p>{pct:.1f}% of total</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Charts ──
    col_bar, col_pie = st.columns(2)

    labels  = list(DATASET_COUNTS.keys())
    values  = list(DATASET_COUNTS.values())
    colors  = [class_colors[c] for c in labels]

    with col_bar:
        st.markdown("#### Distribution by Class")
        fig_bar = go.Figure(go.Bar(
            x=[l.replace("_", " ") for l in labels],
            y=values,
            marker=dict(color=colors, opacity=0.85),
            text=values,
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            xaxis=dict(showgrid=False, tickfont=dict(size=12)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Count"),
            margin=dict(t=10, b=10, l=0, r=0),
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        st.markdown("#### Proportional Split")
        fig_pie = go.Figure(go.Pie(
            labels=[l.replace("_", " ") for l in labels],
            values=values,
            marker=dict(colors=colors, line=dict(color="#0a0e1a", width=2)),
            hole=0.45,
            textinfo="label+percent",
            textfont=dict(size=12, color="#e2e8f0"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Pct: %{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            showlegend=False,
            margin=dict(t=10, b=10, l=0, r=0),
            height=350,
            annotations=[dict(text=f"<b>{total:,}</b><br>Total", x=0.5, y=0.5, font_size=14, showarrow=False, font=dict(color="#e2e8f0"))],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Imbalance note ──
    st.markdown("""
    <div class='info-box'>
        ⚖️ <b>Class Imbalance:</b> The dataset is significantly imbalanced — <b>No_DR</b> has 9.4× more images than <b>Severe</b>. 
        To handle this, <b>sklearn's compute_class_weight</b> was used during training to assign higher loss penalties to minority classes. 
        This helps the model learn better representations for rare but clinically critical grades.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Sample images per class ──
    st.markdown("#### 🖼️ Sample Images per DR Grade")

    for cls, info in SEVERITY_MAP.items():
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:12px; margin: 20px 0 10px 0;'>
            <div style='font-size:1.4rem;'>{info["icon"]}</div>
            <div>
                <div style='font-weight:700; color:{info["color"]}; font-size:1.05rem;'>{cls.replace("_"," ")}</div>
                <div style='font-size:0.8rem; color:#64748b;'>{info["desc"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        samples = get_sample_images(cls, n=4)
        if samples:
            img_cols = st.columns(min(len(samples), 4))
            for j, img in enumerate(samples):
                with img_cols[j]:
                    st.image(img, use_container_width=True)
        else:
            st.info(f"No sample images found in `colored_images/{cls}/`")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 4 — Model Performance
# ─────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.markdown("# 📈 Model Performance")
    st.markdown(
        "<p style='color:#94a3b8; font-size:1rem; margin-top:-12px;'>Evaluation metrics for EfficientNetB3 on the held-out test set (367 samples)</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Key metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Test Accuracy", "77.38%")
    c2.metric("📊 Macro F1", "0.52")
    c3.metric("📊 Weighted F1", "0.74")
    c4.metric("🖼️ Test Samples", "367")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Two columns: table + radar ──
    col_table, col_radar = st.columns([1.1, 1])

    with col_table:
        st.markdown("#### 📋 Classification Report")
        df_report = pd.DataFrame(CLASSIFICATION_REPORT)
        df_report["Support"] = df_report["Support"].astype(int)

        # Colour-coded dataframe
        def color_f1(val):
            if val >= 0.80: return "background-color: rgba(34,197,94,0.25); color:#22c55e"
            elif val >= 0.50: return "background-color: rgba(245,158,11,0.25); color:#f59e0b"
            else: return "background-color: rgba(239,68,68,0.25); color:#ef4444"

        styled = (
            df_report.style
            .format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"})
            .map(color_f1, subset=["Precision", "Recall", "F1-Score"])
            .set_properties(**{"background-color": "rgba(0,0,0,0)", "color": "#e2e8f0", "border": "1px solid rgba(255,255,255,0.08)"})
            .set_table_styles([
                {"selector": "th", "props": [("background-color", "rgba(56,189,248,0.1)"), ("color", "#38bdf8"), ("font-weight", "600"), ("padding", "10px 14px")]},
                {"selector": "td", "props": [("padding", "10px 14px")]},
            ])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Weighted averages note
        st.markdown("""
        <div style='font-size:0.8rem; color:#64748b; margin-top:6px;'>
            Macro avg F1: 0.52 · Weighted avg F1: 0.74 · Support: 367
        </div>
        """, unsafe_allow_html=True)

    with col_radar:
        st.markdown("#### 🕸️ Per-Class F1 Radar")
        classes  = CLASSIFICATION_REPORT["Class"]
        f1_vals  = CLASSIFICATION_REPORT["F1-Score"]
        fig_radar = go.Figure(go.Scatterpolar(
            r=f1_vals + [f1_vals[0]],
            theta=classes + [classes[0]],
            fill="toself",
            fillcolor="rgba(56,189,248,0.12)",
            line=dict(color="#38bdf8", width=2),
            marker=dict(color="#818cf8", size=8),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#94a3b8", size=10), gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(tickfont=dict(color="#cbd5e1", size=12), gridcolor="rgba(255,255,255,0.08)"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(t=20, b=20, l=20, r=20),
            height=360,
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Confusion Matrix ──
    st.markdown("#### 🔢 Confusion Matrix (Test Set)")
    class_labels_cm = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

    fig_cm = go.Figure(go.Heatmap(
        z=CONFUSION_MATRIX,
        x=[f"Pred: {c.replace('_',' ')}" for c in class_labels_cm],
        y=[f"True: {c.replace('_',' ')}" for c in class_labels_cm],
        text=CONFUSION_MATRIX,
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        colorscale=[
            [0.0, "rgba(56,189,248,0.05)"],
            [0.3, "rgba(56,189,248,0.3)"],
            [0.7, "rgba(129,140,248,0.6)"],
            [1.0, "rgba(124,58,237,0.9)"],
        ],
        showscale=True,
        colorbar=dict(tickfont=dict(color="#94a3b8")),
    ))
    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(tickfont=dict(size=12), side="bottom"),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        margin=dict(t=10, b=10, l=0, r=0),
        height=420,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Per-class precision vs recall bars ──
    st.markdown("#### ⚖️ Precision vs Recall by Class")
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Bar(
        name="Precision",
        x=CLASSIFICATION_REPORT["Class"],
        y=CLASSIFICATION_REPORT["Precision"],
        marker_color="rgba(56,189,248,0.75)",
    ))
    fig_pr.add_trace(go.Bar(
        name="Recall",
        x=CLASSIFICATION_REPORT["Class"],
        y=CLASSIFICATION_REPORT["Recall"],
        marker_color="rgba(129,140,248,0.75)",
    ))
    fig_pr.add_trace(go.Bar(
        name="F1-Score",
        x=CLASSIFICATION_REPORT["Class"],
        y=CLASSIFICATION_REPORT["F1-Score"],
        marker_color="rgba(34,197,94,0.75)",
    ))
    fig_pr.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", range=[0, 1.1], title="Score"),
        legend=dict(font=dict(color="#e2e8f0"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=0, r=0),
        height=320,
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("""
    <div class='info-box'>
        💡 <b>Key Observations:</b><br>
        • <b>No_DR</b> has excellent F1 (0.95) — the model reliably identifies healthy retinas.<br>
        • <b>Moderate</b> performs well (F1=0.73) likely due to having more training samples.<br>
        • <b>Mild</b> has the lowest recall (0.05) — frequently misclassified as Moderate, a known challenge in DR grading.<br>
        • <b>Severe & Proliferate_DR</b> are harder to classify due to fewer training samples.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 5 — About
# ─────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Project")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🧠 Model Architecture")
        st.markdown("""
        <div class='info-box'>
            <b>Base Model:</b> EfficientNetB3 (pretrained on ImageNet)<br><br>
            <b>Custom Head:</b><br>
            &nbsp;→ GlobalAveragePooling2D<br>
            &nbsp;→ Dropout(0.4)<br>
            &nbsp;→ Dense(5, activation='softmax')<br><br>
            <b>Training Strategy:</b><br>
            &nbsp;• Phase 1: Base model frozen → train classification head<br>
            &nbsp;• Optimizer: Adam (lr=0.001)<br>
            &nbsp;• Loss: Categorical Cross-Entropy + Class Weights<br>
            &nbsp;• Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau<br><br>
            <b>Input Size:</b> 224 × 224 × 3<br>
            <b>Parameters:</b> ~12M (EfficientNetB3 base)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🖼️ Preprocessing Pipeline")
        st.markdown("""
        <div class='info-box'>
            1. Resize to 224×224<br>
            2. Convert BGR → LAB colour space<br>
            3. Apply CLAHE on L-channel (clipLimit=2.0, tile=8×8)<br>
            4. Convert back to RGB<br>
            5. Data augmentation (train only):<br>
            &nbsp;&nbsp;• Random rotation ±15°<br>
            &nbsp;&nbsp;• Width/height shift 10%<br>
            &nbsp;&nbsp;• Zoom ±15%<br>
            &nbsp;&nbsp;• Horizontal & vertical flip
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### 📚 DR Grade Mapping")
        grade_data = {
            "Code": [0, 1, 2, 3, 4],
            "Class Name": ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"],
            "ICDR Grade": ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"],
            "Clinical Meaning": [
                "No abnormalities",
                "Microaneurysms only",
                ">MA, <Severe NPDR",
                "Extensive IRMA/beading",
                "Neovascularisation",
            ],
        }
        df_grade = pd.DataFrame(grade_data)
        st.dataframe(df_grade, use_container_width=True, hide_index=True)

        st.markdown("### 🔬 Explainability: Grad-CAM")
        st.markdown("""
        <div class='info-box'>
            <b>Gradient-weighted Class Activation Mapping (Grad-CAM)</b> produces visual explanations 
            by computing the gradient of the predicted class score with respect to the last 
            convolutional layer's feature maps.<br><br>
            This highlights <b>which regions of the retina</b> influenced the model's decision — 
            making the system interpretable for clinical use.<br><br>
            Layer used: <code style='background:rgba(56,189,248,0.1); padding:2px 6px; border-radius:4px;'>top_conv</code> 
            (last conv block in EfficientNetB3)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Tech Stack")
        tech = [
            ("🤖", "TensorFlow 2.x / Keras", "Model training & inference"),
            ("📊", "Streamlit", "Dashboard framework"),
            ("📈", "Plotly", "Interactive visualizations"),
            ("👁️", "OpenCV", "Image processing & Grad-CAM"),
            ("🐼", "Pandas / NumPy", "Data manipulation"),
            ("🔬", "scikit-learn", "Metrics & class weights"),
        ]
        for icon, name, role in tech:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05);'>
                <span style='font-size:1.2rem;'>{icon}</span>
                <div>
                    <div style='font-weight:600; color:#e2e8f0; font-size:0.9rem;'>{name}</div>
                    <div style='font-size:0.78rem; color:#64748b;'>{role}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📖 References")
    refs = [
        ("EfficientNet: Rethinking Model Scaling for CNNs", "Tan & Le, ICML 2019", "https://arxiv.org/abs/1905.11946"),
        ("Grad-CAM: Visual Explanations from Deep Networks", "Selvaraju et al., ICCV 2017", "https://arxiv.org/abs/1610.02391"),
        ("APTOS 2019 Blindness Detection Dataset", "Kaggle / Aravind Eye Hospital", "https://www.kaggle.com/c/aptos2019-blindness-detection"),
        ("International Clinical Diabetic Retinopathy Scale", "Wilkinson et al., Ophthalmology 2003", "https://doi.org/10.1016/S0161-6420(03)00475-5"),
    ]
    for title, authors, url in refs:
        st.markdown(f"""
        <div style='padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.05);'>
            <a href='{url}' target='_blank' style='color:#38bdf8; font-weight:600; text-decoration:none;'>{title}</a>
            <div style='font-size:0.82rem; color:#64748b; margin-top:3px;'>{authors}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; padding: 20px 0; color:#475569; font-size:0.85rem;'>
        Built with ❤️ for medical AI research · 
        <span style='color:#38bdf8;'>EfficientNetB3</span> · 
        <span style='color:#818cf8;'>Streamlit</span> · 
        <span style='color:#22c55e;'>TensorFlow</span>
        <br><br>
        ⚠️ For research and educational purposes only. Not a substitute for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)
