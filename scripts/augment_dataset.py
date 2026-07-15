import os
import traceback

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array
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
# DATASET PATHS
# ============================================================

INPUT_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "processed",
    "ginger"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "augmented",
    "ginger"
)

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)

# ============================================================
# VALID IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)

AUGMENTATIONS_PER_IMAGE = 5

# ============================================================
# IMAGE AUGMENTOR
# ============================================================

augmentor = ImageDataGenerator(

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.2,

    shear_range=0.2,

    horizontal_flip=True,

    vertical_flip=False,

    brightness_range=[0.8, 1.2],

    fill_mode='nearest'
)

# ============================================================
# VERIFY INPUT PATH
# ============================================================

if not os.path.exists(INPUT_PATH):

    raise FileNotFoundError(
        f"\nProcessed Ginger Dataset Not Found:\n{INPUT_PATH}"
    )

# ============================================================
# START AUGMENTATION
# ============================================================

print("\nStarting Dataset Augmentation...\n")

total_processed = 0
total_generated = 0

# ============================================================
# CLASS LOOP
# ============================================================

for class_name in os.listdir(INPUT_PATH):

    class_input_path = os.path.join(
        INPUT_PATH,
        class_name
    )

    if not os.path.isdir(class_input_path):
        continue

    class_output_path = os.path.join(
        OUTPUT_PATH,
        class_name
    )

    os.makedirs(
        class_output_path,
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print(f"Processing Class: {class_name}")
    print("=" * 60)

    # ========================================================
    # IMAGE LOOP
    # ========================================================

    for image_name in os.listdir(class_input_path):

        if not image_name.lower().endswith(
            VALID_EXTENSIONS
        ):
            continue

        image_path = os.path.join(
            class_input_path,
            image_name
        )

        try:

            # =================================================
            # LOAD IMAGE
            # =================================================

            image = load_img(
                image_path,
                target_size=IMG_SIZE
            )

            image_array = img_to_array(image)

            image_array = image_array.reshape(
                (1,) + image_array.shape
            )

            save_prefix = os.path.splitext(
                image_name
            )[0]

            # =================================================
            # GENERATE AUGMENTED IMAGES
            # =================================================

            generated_count = 0

            for batch in augmentor.flow(

                image_array,

                batch_size=1,

                save_to_dir=class_output_path,

                save_prefix=save_prefix,

                save_format='jpg'
            ):

                generated_count += 1
                total_generated += 1

                if generated_count >= AUGMENTATIONS_PER_IMAGE:
                    break

            total_processed += 1

            print(
                f"Augmented: {image_name} "
                f"-> Generated {generated_count} Images"
            )

        except Exception as error:

            print("\nERROR PROCESSING IMAGE")
            print(f"Image: {image_name}")
            print(f"Path: {image_path}")

            print("\nException:")
            print(error)

            traceback.print_exc()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("AUGMENTATION COMPLETE")
print("=" * 60)

print(f"\nOriginal Images Processed: {total_processed}")

print(f"Total Augmented Images Generated: {total_generated}")

print(f"\nAugmented Dataset Saved At:\n{OUTPUT_PATH}")

print("\nDataset Augmentation Finished Successfully.\n")