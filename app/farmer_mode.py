import os
import json
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image

import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)

from src.morphology.morphology_engine import MorphologyEngine
from src.fusion.bi_phasic_logic import BiPhasicLogic
from src.voice.voice_engine import VoiceEngine

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Rhizo-Net Farmer Mode",
    layout="centered"
)

# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH)

# ============================================================
# LOAD ENGINES
# ============================================================

morphology_engine = MorphologyEngine()

logic_engine = BiPhasicLogic()

voice_engine = VoiceEngine()

# ============================================================
# CLASS LABELS
# ============================================================

CLASS_NAMES = [
    "Bacterial Wilt",
    "Healthy",
    "Pythium Soft Rot"
]

# ============================================================
# LOCALIZATION
# ============================================================

LANGUAGES = {
    "English": "en.json",
    "Hindi": "hi.json",
    "Marathi": "mr.json"
}

selected_language = st.sidebar.selectbox(
    "Select Language",
    list(LANGUAGES.keys())
)

language_file = os.path.join(
    BASE_DIR,
    "app",
    "localization",
    LANGUAGES[selected_language]
)

with open(language_file, "r", encoding="utf-8") as file:
    texts = json.load(file)

# ============================================================
# UI HEADER
# ============================================================

st.title(texts["title"])

st.markdown("""
### 🌿 AI-based Rhizome Disease Detection System
""")

# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    texts["upload"],
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# INFERENCE
# ============================================================

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Leaf Image",
        width=400
    )

    image_np = np.array(image)

    resized = cv2.resize(image_np, (224, 224))

    normalized = resized / 255.0

    prediction = model.predict(
        np.expand_dims(normalized, axis=0)
    )

    confidence = float(np.max(prediction))

    predicted_class = CLASS_NAMES[
        np.argmax(prediction)
    ]

    # ========================================================
    # TEMP IMAGE SAVE
    # ========================================================

    temp_path = os.path.join(
        BASE_DIR,
        "temp.jpg"
    )

    cv2.imwrite(temp_path, resized)

    # ========================================================
    # MORPHOLOGY ANALYSIS
    # ========================================================

    edge_density = morphology_engine.calculate_edge_density(
        temp_path
    )

    # ========================================================
    # LOGIC FUSION
    # ========================================================

    def localize_result(result, texts):

        diagnosis_map = {

            "Healthy Crop":
                texts["healthy_crop"],

            "Early Foliar Infection":
                texts["early_infection"],

            "Advanced Pythium Soft Rot":
                texts["advanced_rot"],

            "Green Wilt / Drought Stress":
                texts["green_wilt"],

            "Blotch":
                texts["blotch"],

            "Leaf Spot":
                texts["leaf_spot"],
        }

        action_map = {

            "No intervention required.":
                texts["healthy_action"],

            "Apply preventive fungicide.":
                texts["early_action"],

            "Quarantine infected crop.":
                texts["advanced_action"],

            "Check irrigation immediately.":
                texts["green_action"],

            "Apply copper-based fungicide and isolate affected leaves.":
                texts["blotch_action"],

            "Remove infected leaves and apply preventive treatment.":
                texts["leaf_spot_action"],
        }

        result["diagnosis"] = diagnosis_map.get(
            result["diagnosis"],
            result["diagnosis"]
        )

        result["action"] = action_map.get(
            result["action"],
            result["action"]
        )

        return result

    result = logic_engine.diagnose(
        predicted_class,
        confidence,
        edge_density
    )

    result = localize_result(
        result,
        texts
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.success("Analysis Complete")

    st.subheader("Spectral Classification")

    st.write(predicted_class)

    st.subheader("Confidence Score")

    st.write(f"{confidence:.2f}")

    st.subheader("Leaf Edge Density")

    st.write(f"{edge_density:.2f}")

    st.subheader("Final Diagnosis")

    st.error(result["diagnosis"])

    st.subheader("Farmer Recommendation")

    st.info(result["action"])

    # ========================================================
    # VOICE OUTPUT
    # ========================================================

    if st.button(texts["voice_button"]):

        with st.spinner("Generating Voice Guidance..."):

            voice_engine.speak_slowly(
                result["diagnosis"],
                result["action"]
            )

        st.success(
            texts["voice_done"]
        )