"""Central configuration: brand, models, paths and shared constants."""
import os

from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ (won't overwrite existing vars)

# ---- Brand -------------------------------------------------------------------------
STORE_NAME_TA = "யாழ் சூப்பர் ஸ்டோர்ஸ்"
STORE_NAME_EN = "Yaazh Super Stores"
TAGLINE_TA = "ஒவ்வொரு வீட்டின் நம்பிக்கை"
TAGLINE_EN = "The trust of every home"
STORE_NAME = f"{STORE_NAME_TA} ({STORE_NAME_EN})"

# ---- Storage -----------------------------------------------------------------------
DB_PATH = os.getenv("SUPPORT_DB", "support.db")
FAISS_DIR = os.getenv("FAISS_DIR", "./faiss_db")
KB_DIR = os.getenv("KB_DIR", "./kb_docs")   # folder of PDF/txt/md manuals that get embedded

# ---- Models ------------------------------------------------------------------------
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")              # must be vision-capable
FAST_MODEL = os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini")    # cheaper agents (triage, warranty, QA)
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# ---- Domain constants --------------------------------------------------------------
CATEGORIES = ["Not Working / Power", "Cooling / Heating", "Water Leak / Drain",
              "Noise / Vibration", "Error Code", "Warranty / Repair", "Installation",
              "Billing / Return", "Other"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]
PRODUCTS = ["Washing Machine", "Refrigerator", "Air Conditioner", "Dishwasher",
            "Microwave Oven", "Oven / Cooktop", "Water Purifier", "Vacuum Cleaner"]

# Shared by every customer-facing / reasoning agent prompt.
IDENTITY = (f"You work for {STORE_NAME_EN} ({STORE_NAME_TA}), an appliance store whose tagline is "
            f"'{TAGLINE_EN}' ({TAGLINE_TA}).")
LANGUAGE_RULE = (
    "Respond in English by default, regardless of the language the customer writes in. Change the response "
    "language only when the customer explicitly asks for a different language; then use that language until "
    "the customer explicitly requests another change."
)
