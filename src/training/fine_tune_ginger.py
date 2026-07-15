import os
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
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

GINGER_DATASET = os.path.join(
    BASE_DIR,
    "datasets",
    "augmented",
    "ginger"
)

# ============================================================
# PRETRAINED MODEL PATH
# ============================================================

PRETRAINED_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "checkpoints",
    "maize_pretrained.h5"
)

# ============================================================
# FINAL MODEL PATH
# ============================================================

FINAL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "rhizonet_final.h5"
)

# ============================================================
# CREATE SAVE DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(FINAL_MODEL_PATH),
    exist_ok=True
)

# ============================================================
# VERIFY PATHS
# ============================================================

if not os.path.exists(GINGER_DATASET):

    raise FileNotFoundError(
        f"\nGinger Dataset Not Found:\n{GINGER_DATASET}"
    )

if not os.path.exists(PRETRAINED_MODEL):

    raise FileNotFoundError(
        f"\nPretrained Model Not Found:\n{PRETRAINED_MODEL}"
    )

print("\nDataset and Model Paths Verified.\n")

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

EPOCHS = 15

# ============================================================
# DATA GENERATOR
# ============================================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2
)

# ============================================================
# TRAINING DATA
# ============================================================

train_data = train_datagen.flow_from_directory(

    GINGER_DATASET,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='training'
)

# ============================================================
# VALIDATION DATA
# ============================================================

val_data = train_datagen.flow_from_directory(

    GINGER_DATASET,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='validation'
)

# ============================================================
# SHOW DETECTED CLASSES
# ============================================================

print("\nDetected Ginger Classes:\n")

print(train_data.class_indices)

NUM_CLASSES = train_data.num_classes

print(f"\nNumber of Classes: {NUM_CLASSES}\n")

# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

print("\nLoading Pretrained Maize Model...\n")

pretrained_model = load_model(PRETRAINED_MODEL)

# ============================================================
# REMOVE OLD CLASSIFICATION HEAD
# ============================================================

base_model = pretrained_model.layers[0]

# ============================================================
# FREEZE LOWER LAYERS
# ============================================================

for layer in base_model.layers[:-20]:

    layer.trainable = False

# ============================================================
# UNFREEZE UPPER LAYERS
# ============================================================

for layer in base_model.layers[-20:]:

    layer.trainable = True

# ============================================================
# BUILD NEW GINGER MODEL
# ============================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation='relu'
)(x)

x = Dropout(0.4)(x)

output = Dense(
    NUM_CLASSES,
    activation='softmax'
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=Adam(learning_rate=1e-5),

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    FINAL_MODEL_PATH,

    monitor='val_accuracy',

    save_best_only=True,

    verbose=1
)

early_stop = EarlyStopping(

    monitor='val_accuracy',

    patience=5,

    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.2,

    patience=2,

    verbose=1
)

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nFinal Rhizo-Net Architecture:\n")

model.summary()

# ============================================================
# START FINE-TUNING
# ============================================================

print("\nStarting Ginger Fine-Tuning...\n")

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(FINAL_MODEL_PATH)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("GINGER FINE-TUNING COMPLETE")
print("=" * 60)

print(f"\nFinal Model Saved At:\n{FINAL_MODEL_PATH}")

print("\nRhizo-Net Model Ready.\n")