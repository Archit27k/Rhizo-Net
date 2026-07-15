import os
import tensorflow as tf

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
# TFLITE OUTPUT PATH
# ============================================================

TFLITE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tflite",
    "rhizonet.tflite"
)

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(TFLITE_PATH),
    exist_ok=True
)

# ============================================================
# VERIFY MODEL EXISTS
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nTrained Model Not Found:\n{MODEL_PATH}"
    )

print("\nLoading Trained Rhizo-Net Model...\n")

# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel Loaded Successfully.\n")

# ============================================================
# INITIALIZE TFLITE CONVERTER
# ============================================================

print("\nInitializing TensorFlow Lite Converter...\n")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

# ============================================================
# OPTIMIZATION
# ============================================================

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

# ============================================================
# OPTIONAL FP16 QUANTIZATION
# ============================================================

converter.target_spec.supported_types = [
    tf.float16
]

# ============================================================
# CONVERT MODEL
# ============================================================

print("\nConverting Model To TensorFlow Lite...\n")

tflite_model = converter.convert()

# ============================================================
# SAVE TFLITE MODEL
# ============================================================

with open(TFLITE_PATH, "wb") as file:

    file.write(tflite_model)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("TENSORFLOW LITE CONVERSION COMPLETE")
print("=" * 60)

print(f"\nTFLite Model Saved At:\n{TFLITE_PATH}")

print("\nRhizo-Net Edge Model Ready.\n")