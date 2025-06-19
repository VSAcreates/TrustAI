import os
from pathlib import Path
from datetime import datetime, timedelta

# Project directories
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(BASE_DIR, "data")
GENERATED_DIR = os.path.join(DATA_DIR, "generated")
PATTERNS_DIR = os.path.join(DATA_DIR, "patterns")

# File paths
TRANSACTIONS_FILE = os.path.join(GENERATED_DIR, "transactions.csv")
IDENTITIES_FILE = os.path.join(GENERATED_DIR, "identities.csv")
PATTERNS_FILE = os.path.join(PATTERNS_DIR, "fraud_patterns.json")

# Create directories if they don't exist
Path(GENERATED_DIR).mkdir(parents=True, exist_ok=True)
Path(PATTERNS_DIR).mkdir(parents=True, exist_ok=True)

# Simulation parameters
NUM_USERS = 1000
NUM_TRANSACTIONS = 10000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 6, 30)

# Fraud patterns to simulate
FRAUD_PATTERNS = {
    "high_value_rapid": {
        "description": "Multiple high-value transactions in short time",
        "threshold_amount": 5000,
        "time_window": timedelta(hours=1),
        "min_transactions": 3
    },
    "geo_velocity": {
        "description": "Impossible geographic movement between transactions",
        "max_realistic_speed_kmh": 800  # Maximum realistic travel speed
    },
    "new_device_high_value": {
        "description": "High value transaction from new device",
        "threshold_amount": 3000,
        "device_age_days": 1
    }
}