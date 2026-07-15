import os
import shutil

# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ============================================================
# SOURCE PATHS
# ============================================================

RAW_GINGER = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "ginger"
)

RAW_MAIZE = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "maize"
)

# ============================================================
# DESTINATION PATHS
# ============================================================

PROC_GINGER = os.path.join(
    BASE_DIR,
    "datasets",
    "processed",
    "ginger"
)

PROC_MAIZE = os.path.join(
    BASE_DIR,
    "datasets",
    "processed",
    "maize"
)

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(PROC_GINGER, exist_ok=True)
os.makedirs(PROC_MAIZE, exist_ok=True)

# ============================================================
# VERIFY INPUT DIRECTORIES
# ============================================================

if not os.path.exists(RAW_GINGER):

    raise FileNotFoundError(
        f"Ginger dataset not found:\n{RAW_GINGER}"
    )

if not os.path.exists(RAW_MAIZE):

    raise FileNotFoundError(
        f"Maize dataset not found:\n{RAW_MAIZE}"
    )

# ============================================================
# COPY GINGER DATASET
# ============================================================

print("\nOrganizing Ginger Dataset...\n")

for folder in os.listdir(RAW_GINGER):

    src = os.path.join(RAW_GINGER, folder)

    dst = os.path.join(PROC_GINGER, folder)

    if os.path.isdir(src):

        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True
        )

        print(f"Copied Ginger Class: {folder}")

# ============================================================
# COPY MAIZE DATASET
# ============================================================

print("\nOrganizing Maize Dataset...\n")

for folder in os.listdir(RAW_MAIZE):

    src = os.path.join(RAW_MAIZE, folder)

    dst = os.path.join(PROC_MAIZE, folder)

    if os.path.isdir(src):

        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True
        )

        print(f"Copied Maize Class: {folder}")

print("\nDataset Organization Complete.\n")