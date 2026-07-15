import os

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

RAW_DATASET = os.path.join(
    BASE_DIR,
    "datasets",
    "raw"
)

PROCESSED_DATASET = os.path.join(
    BASE_DIR,
    "datasets",
    "processed"
)

AUGMENTED_DATASET = os.path.join(
    BASE_DIR,
    "datasets",
    "augmented"
)

# ============================================================
# VERIFY FUNCTION
# ============================================================

def verify_directory(path, title):

    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)

    if not os.path.exists(path):

        print(f"\nDirectory NOT Found:\n{path}\n")
        return

    total_images = 0

    for root, dirs, files in os.walk(path):

        image_files = [

            file for file in files

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )
        ]

        if len(image_files) > 0:

            relative_path = os.path.relpath(root, path)

            print(f"\nClass Folder: {relative_path}")

            print(f"Images Found: {len(image_files)}")

            total_images += len(image_files)

    print("\n" + "-" * 60)

    print(f"Total Images: {total_images}")

    print("-" * 60)

# ============================================================
# EXECUTION
# ============================================================

verify_directory(
    RAW_DATASET,
    "RAW DATASET VERIFICATION"
)

verify_directory(
    PROCESSED_DATASET,
    "PROCESSED DATASET VERIFICATION"
)

verify_directory(
    AUGMENTED_DATASET,
    "AUGMENTED DATASET VERIFICATION"
)

print("\nDataset Verification Complete.\n")