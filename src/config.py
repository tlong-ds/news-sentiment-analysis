import os
from pathlib import Path

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Main work directory (CafeF and VN-Index prices)
MAIN_DATA_DIR = os.path.join(DATA_DIR, "main")
RAW_DATA_DIR = os.path.join(MAIN_DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(MAIN_DATA_DIR, "processed")
CAFEF_DATA_DIR = os.path.join(MAIN_DATA_DIR, "cafef")
MODELS_DATA_DIR = os.path.join(BASE_DIR, "models")


# ViFiC specific directory (unsupervised domain pretraining, dual-LLM annotation)
VIFIC_DATA_DIR = os.path.join(DATA_DIR, "vific")
FINETUNES_DATA_DIR = os.path.join(VIFIC_DATA_DIR, "fine-tunes")
VIFIC_NORMALIZED_DIR = os.path.join(VIFIC_DATA_DIR, "processed")
ANNOTATION_DATA_DIR = os.path.join(VIFIC_DATA_DIR, "annotation")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CAFEF_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DATA_DIR, exist_ok=True)
os.makedirs(FINETUNES_DATA_DIR, exist_ok=True)
os.makedirs(VIFIC_NORMALIZED_DIR, exist_ok=True)
os.makedirs(ANNOTATION_DATA_DIR, exist_ok=True)


# --- Collection Configuration ---
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

REQUEST_DELAY_SECONDS = 0.6
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# --- Business-domain keywords (shared across sources) ---
BUSINESS_KEYWORDS = {
    "chung-khoan": "Chứng khoán",
    "thi-truong-chung-khoan": "Chứng khoán",
    "doanh-nghiep": "Doanh nghiệp",
    "tai-chinh": "Tài chính",
    "tai-chinh-ngan-hang": "Tài chính ngân hàng",
    "kinh-te": "Kinh tế",
    "kinh-te-vi-mo": "Vĩ mô",
    "vi-mo": "Vĩ mô",
    "bat-dong-san": "Bất động sản",
    "kinh-doanh": "Kinh doanh",
}

# --- CafeF ---
CAFEF_BASE_URL = "https://cafef.vn"
CAFEF_CATEGORIES = {
    "thi-truong-chung-khoan": {"name": "Chứng khoán", "id": 31},
    "bat-dong-san": {"name": "Bất động sản", "id": 35},
    "doanh-nghiep": {"name": "Doanh nghiệp", "id": 36},
    "tai-chinh-ngan-hang": {"name": "Tài chính ngân hàng", "id": 34},
    "kinh-te-vi-mo-dau-tu": {"name": "Vĩ mô", "id": 33},
}

# --- vnstock ---
# Major VN-Index constituents for ticker-level news collection.
# Covers banking, real estate, tech, consumer, energy — the sectors
# most relevant for financial sentiment analysis.
VNSTOCK_SYMBOLS = [
    "VNM", "VCB", "VIC", "VHM", "HPG",
    "FPT", "MBB", "TCB", "CTG", "BID",
    "MSN", "VNR", "SSI", "VND", "HCM",
    "PLX", "GAS", "SAB", "MWG", "PNJ",
]
VNSTOCK_PROVIDER_ORDER = ["VCI", "KBS"]
VNSTOCK_PAGE_SIZE = 50
VNSTOCK_MAX_PAGES = 10

# --- ViFiC (Vietnamese Financial Corpus) ---
# Download from Kaggle: https://www.kaggle.com/datasets/...
# Place the extracted files under data/fine-tunes/
VIFIC_DATA_DIR = os.path.join(FINETUNES_DATA_DIR)

# --- VnCoreNLP Configuration ---
VNCORENLP_JAR_PATH = os.environ.get(
    "VNCORENLP_JAR_PATH",
    os.path.join(BASE_DIR, "vncorenlp", "VnCoreNLP-1.1.1.jar")
)

# --- Output files ---
SOURCE_OUTPUTS = {
    "cafef": "news_VN_cafef.csv",
    "vnstock": "news_VN_vnstock.csv",
}


