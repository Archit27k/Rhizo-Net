import os
import sys
import cv2
import time
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image

# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)

# ============================================================
# IMPORT ENGINES
# ============================================================

from src.morphology.morphology_engine import MorphologyEngine
from src.fusion.bi_phasic_logic import BiPhasicLogic

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Rhizo-Net",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0b1220;
        color: white;
    }

    .stApp {
        background: linear-gradient(
            135deg,
            #0b1220,
            #111827,
            #0f172a
        );
    }

    h1 {
        color: white !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        color: #d1fae5 !important;
    }

    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
        text-align: center;
    }

    .diagnosis-box {
        padding: 20px;
        border-radius: 16px;
        background: linear-gradient(135deg,#14532d,#166534);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }

    .recommendation-box {
        padding: 20px;
        border-radius: 16px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04);
        border-radius: 20px;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "rhizonet_final.h5"
)

# ============================================================
# FAST MODEL LOADING
# ============================================================

@st.cache_resource

def load_model_cached():

    return tf.keras.models.load_model(MODEL_PATH)

# ============================================================
# LOAD MODEL ONCE
# ============================================================

model = load_model_cached()

# ============================================================
# LOAD ENGINES
# ============================================================

morph_engine = MorphologyEngine()
logic_engine = BiPhasicLogic()

# ============================================================
# CLASS LABELS
# ============================================================

CLASS_NAMES = [
    "Bacterial Wilt",
    "Healthy",
    "Pythium Soft Rot"
]

# ============================================================
# HEADER
# ============================================================

col_logo, col_title = st.columns([1, 8])

with col_logo:
    st.markdown("# 🌿")

with col_title:
    st.title("Rhizo-Net")
    st.caption(
        "AI-Powered Rhizome Disease Detection System"
    )

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload Ginger Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# IMAGE PROCESSING FUNCTION
# ============================================================

@st.cache_data

def preprocess_image(_image):

    image_array = np.array(_image)

    if len(image_array.shape) == 2:

        image_array = cv2.cvtColor(
            image_array,
            cv2.COLOR_GRAY2RGB
        )

    height, width = image_array.shape[:2]

    scale = 224 / max(height, width)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image_array,
        (new_width, new_height)
    )

    padded = np.zeros(
        (224, 224, 3),
        dtype=np.uint8
    )

    x_offset = (224 - new_width) // 2
    y_offset = (224 - new_height) // 2

    padded[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    normalized = padded / 255.0

    return padded, normalized

# ============================================================
# INFERENCE
# ============================================================

if uploaded:

    start_time = time.time()

    image = Image.open(uploaded)

    col1, col2 = st.columns([1.2, 1])

    with col1:

        st.image(
            image,
            caption="Uploaded Leaf Image",
            width=500
        )

    padded, normalized = preprocess_image(image)

    prediction = model.predict(
        np.expand_dims(normalized, axis=0),
        verbose=0
    )

    confidence = float(np.max(prediction))

    predicted_class = CLASS_NAMES[
        np.argmax(prediction)
    ]

    temp_path = os.path.join(
        BASE_DIR,
        "temp.jpg"
    )

    cv2.imwrite(temp_path, padded)

    edge_density = morph_engine.calculate_edge_density(
        temp_path
    )

    result = logic_engine.diagnose(
        predicted_class,
        confidence,
        edge_density
    )

    inference_time = time.time() - start_time

    # ========================================================
    # RIGHT PANEL
    # ========================================================

    with col2:

        st.markdown(
            f"""
            <div class='diagnosis-box'>
            {result['diagnosis']}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        metric1, metric2 = st.columns(2)

        with metric1:
            st.markdown(
                f"""
                <div class='metric-card'>
                <h3>Confidence</h3>
                <h2>{confidence:.2f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metric2:
            st.markdown(
                f"""
                <div class='metric-card'>
                <h3>Edge Density</h3>
                <h2>{edge_density:.2f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class='recommendation-box'>
            <h3>Recommendation</h3>
            <p>{result['action']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.success(
            f"Inference Time: {inference_time:.2f} sec"
        )

    # ========================================================
    # PROBABILITY BARS
    # ========================================================

    st.subheader("Prediction Confidence Distribution")

    probabilities = prediction[0]

    for idx, class_name in enumerate(CLASS_NAMES):

        st.progress(float(probabilities[idx]))

        st.write(
            f"{class_name}: {probabilities[idx]:.2%}"
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Rhizo-Net • Transfer Learning • MobileNetV2 • Edge AI"
)
