import os
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping
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

# ============================================================
# DATASET PATH
# ============================================================

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "processed",
    "maize"
)

# ============================================================
# MODEL SAVE PATH
# ============================================================

MODEL_SAVE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "checkpoints",
    "maize_pretrained.h5"
)

# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_SAVE_PATH),
    exist_ok=True
)

# ============================================================
# VERIFY DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):

    raise FileNotFoundError(
        f"\nMaize Dataset Not Found:\n{TRAIN_DIR}"
    )

print("\nDataset Path Verified.\n")

# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 10

# ============================================================
# DATA GENERATOR
# ============================================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2
)

# ============================================================
# TRAIN DATA
# ============================================================

train_data = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='training'
)

# ============================================================
# VALIDATION DATA
# ============================================================

val_data = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='validation'
)

# ============================================================
# LOAD MOBILENETV2
# ============================================================

base_model = MobileNetV2(

    weights='imagenet',

    include_top=False,

    input_shape=(224, 224, 3)
)

# ============================================================
# FREEZE LOWER LAYERS
# ============================================================

for layer in base_model.layers:

    layer.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),

    Dropout(0.4),

    Dense(
        train_data.num_classes,
        activation='softmax'
    )
])

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    MODEL_SAVE_PATH,

    monitor='val_accuracy',

    save_best_only=True,

    verbose=1
)

early_stop = EarlyStopping(

    patience=3,

    restore_best_weights=True
)

# ============================================================
# TRAINING
# ============================================================

print("\nStarting Maize Pretraining...\n")

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stop
    ]
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(MODEL_SAVE_PATH)

print("\nMaize Pretraining Complete.\n")

print(f"\nModel Saved At:\n{MODEL_SAVE_PATH}")