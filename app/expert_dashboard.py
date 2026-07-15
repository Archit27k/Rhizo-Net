import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Rhizo-Net Expert Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a,
            #111827,
            #1e293b
        );
        color: white;
    }

    h1, h2, h3 {
        color: white !important;
    }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0px 4px 25px rgba(0,0,0,0.3);
    }

    .glass-panel {
        background: rgba(255,255,255,0.04);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.06);
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
# DATASET PATH
# ============================================================

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "augmented",
    "ginger"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_cached():

    return tf.keras.models.load_model(
        MODEL_PATH
    )

model = load_model_cached()

# ============================================================
# HEADER
# ============================================================

st.title("📊 Rhizo-Net Expert Analytics Dashboard")

st.caption("""
Advanced AI Disease Analytics, Transfer Learning Monitoring,
and Morphological Fusion Evaluation
""")

# ============================================================
# DATASET GENERATOR
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

datagen = ImageDataGenerator(
    rescale=1./255
)

data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ============================================================
# INFERENCE
# ============================================================

start_time = time.time()

predictions = model.predict(
    data,
    verbose=0
)

end_time = time.time()

inference_time = end_time - start_time

# ============================================================
# PREDICTIONS
# ============================================================

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = data.classes

class_labels = list(
    data.class_indices.keys()
)

# ============================================================
# METRICS
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    output_dict=True
)

accuracy = report["accuracy"]

precision = np.mean([
    report[label]["precision"]
    for label in class_labels
])

recall = np.mean([
    report[label]["recall"]
    for label in class_labels
])

f1_score = np.mean([
    report[label]["f1-score"]
    for label in class_labels
])

# ============================================================
# METRIC CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class='metric-card'>
        <h3>Accuracy</h3>
        <h1>{accuracy:.2%}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class='metric-card'>
        <h3>Precision</h3>
        <h1>{precision:.2%}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class='metric-card'>
        <h3>Recall</h3>
        <h1>{recall:.2%}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        f"""
        <div class='metric-card'>
        <h3>F1 Score</h3>
        <h1>{f1_score:.2%}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# SECONDARY METRICS
# ============================================================

col5, col6, col7 = st.columns(3)

with col5:
    st.info(
        f"Inference Time: {inference_time:.2f} sec"
    )

with col6:
    st.info(
        f"Dataset Size: {len(y_true)} Images"
    )

with col7:

    gpu_available = tf.config.list_physical_devices('GPU')

    if gpu_available:
        st.success("GPU Enabled")
    else:
        st.warning("Running on CPU")

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown("---")

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_true,
    y_pred
)

fig_cm, ax_cm = plt.subplots(
    figsize=(8, 8)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_labels
)

disp.plot(ax=ax_cm)

st.pyplot(fig_cm)

# ============================================================
# CLASSIFICATION REPORT TABLE
# ============================================================

st.markdown("---")

st.subheader("Classification Report")

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)

# ============================================================
# ROC CURVES
# ============================================================

st.markdown("---")

st.subheader("ROC Curves")

y_true_bin = label_binarize(
    y_true,
    classes=range(len(class_labels))
)

fig_roc, ax_roc = plt.subplots(
    figsize=(10, 7)
)

for i in range(len(class_labels)):

    fpr, tpr, _ = roc_curve(
        y_true_bin[:, i],
        predictions[:, i]
    )

    roc_auc = auc(fpr, tpr)

    ax_roc.plot(
        fpr,
        tpr,
        label=f"{class_labels[i]} (AUC = {roc_auc:.2f})"
    )

ax_roc.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend()

st.pyplot(fig_roc)

# ============================================================
# PREDICTION CONFIDENCE DISTRIBUTION
# ============================================================

st.markdown("---")

st.subheader("Prediction Confidence Distribution")

confidence_scores = np.max(
    predictions,
    axis=1
)

fig_conf, ax_conf = plt.subplots(
    figsize=(10, 4)
)

ax_conf.hist(
    confidence_scores,
    bins=20
)

ax_conf.set_xlabel("Confidence")
ax_conf.set_ylabel("Frequency")

st.pyplot(fig_conf)

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

st.markdown("---")

st.subheader("Sample Predictions")

sample_images = []

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(
            (".jpg", ".jpeg", ".png")
        ):

            sample_images.append(
                os.path.join(root, file)
            )

sample_images = sample_images[:6]

cols = st.columns(3)

for idx, image_path in enumerate(sample_images):

    image = cv2.imread(image_path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_resized = cv2.resize(
        image,
        (224, 224)
    )

    normalized = image_resized / 255.0

    pred = model.predict(
        np.expand_dims(normalized, axis=0),
        verbose=0
    )

    pred_class = class_labels[
        np.argmax(pred)
    ]

    confidence = np.max(pred)

    with cols[idx % 3]:

        st.image(
            image,
            width=250
        )

        st.success(
            f"{pred_class}"
        )

        st.info(
            f"Confidence: {confidence:.2f}"
        )

# ============================================================
# TRANSFER LEARNING ANALYTICS
# ============================================================

st.markdown("---")

st.subheader("Transfer Learning Analytics")

st.markdown("""
### Model Architecture

- Base Model: MobileNetV2
- Source Domain: Maize Disease Dataset
- Target Domain: Ginger Disease Dataset
- Transfer Learning: Enabled
- Lower Layers Frozen: Yes
- Upper Layers Fine-Tuned: Yes
- Morphological Fusion: Enabled
- TensorFlow Lite Conversion: Enabled
- Streamlit Deployment: Enabled
- Edge AI Compatibility: Enabled
""")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption("""
Rhizo-Net Expert Dashboard • Transfer Learning • MobileNetV2 • Explainable AI
""")