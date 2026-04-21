import os

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Collection Configuration ---
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# --- Scraping Configuration (VnExpress) ---
VNEXPRESS_CATEGORIES = [
    "news/business/economy",
    "news/business/companies",
    "news/business/markets"
]
VNEXPRESS_BASE_URL = "https://e.vnexpress.net"
