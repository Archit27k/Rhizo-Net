import os
import zipfile

from dotenv import load_dotenv

# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ============================================================
# LOAD ENV VARIABLES
# ============================================================

load_dotenv()

username = os.getenv("KAGGLE_USERNAME")
key = os.getenv("KAGGLE_KEY")

os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = key

# ============================================================
# PATHS
# ============================================================

RAW_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "raw"
)

ZIP_PATH = os.path.join(
    RAW_PATH,
    "corn-or-maize-leaf-disease-dataset.zip"
)

EXTRACT_PATH = os.path.join(
    RAW_PATH,
    "maize"
)

# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    RAW_PATH,
    exist_ok=True
)

os.makedirs(
    EXTRACT_PATH,
    exist_ok=True
)

# ============================================================
# DOWNLOAD DATASET
# ============================================================

print("\nDownloading Maize Dataset...\n")

download_command = (

    "kaggle datasets download "

    "-d smaranjitghose/corn-or-maize-leaf-disease-dataset "

    f"-p \"{RAW_PATH}\""
)

download_status = os.system(download_command)

# ============================================================
# VERIFY DOWNLOAD
# ============================================================

if download_status != 0:

    raise RuntimeError(
        "\nKaggle Download Failed.\n"
    )

if not os.path.exists(ZIP_PATH):

    raise FileNotFoundError(
        f"\nZIP File Not Found:\n{ZIP_PATH}"
    )

print("\nDataset Downloaded Successfully.\n")

# ============================================================
# EXTRACT ZIP
# ============================================================

print("\nExtracting Dataset...\n")

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:

    zip_ref.extractall(EXTRACT_PATH)

print("\nExtraction Complete.\n")

# ============================================================
# SHOW EXTRACTED CONTENTS
# ============================================================

print("\nExtracted Classes:\n")

for item in os.listdir(EXTRACT_PATH):

    print(item)

print("\nMaize Dataset Ready.\n")