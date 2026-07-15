import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
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
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.append(BASE_DIR)

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
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# VERIFY PATHS
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel Not Found:\n{MODEL_PATH}"
    )

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset Not Found:\n{DATASET_PATH}"
    )

print("\nLoading Rhizo-Net Model...\n")

# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)

# ============================================================
# DATA GENERATOR
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

print("\nRunning Evaluation...\n")

start_time = time.time()

predictions = model.predict(
    data,
    verbose=1
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
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_csv_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.csv"
)

report_df.to_csv(
    report_csv_path
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

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

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(cm_path)

# ============================================================
# ROC CURVES
# ============================================================

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
ax_roc.set_title("ROC Curves")
ax_roc.legend()

roc_path = os.path.join(
    OUTPUT_DIR,
    "roc_curves.png"
)

plt.savefig(roc_path)

# ============================================================
# METRICS
# ============================================================

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
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("RHIZO-NET EVALUATION COMPLETE")
print("=" * 60)

print(f"\nAccuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-Score      : {f1_score:.4f}")
print(f"Inference Time: {inference_time:.2f} sec")

print("\nGenerated Outputs:")

print(f"\nClassification Report:")
print(report_csv_path)

print(f"\nConfusion Matrix:")
print(cm_path)

print(f"\nROC Curves:")
print(roc_path)

print("\nEvaluation Complete.\n")