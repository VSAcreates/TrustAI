import random
import time
import json
import datetime
import numpy as np
import uuid
from faker import Faker

fake = Faker()

# Constants for simulation
USERS = [
    {"id": i, "name": fake.name(), "country": fake.country(), "card_type": random.choice(["Visa", "Mastercard", "Amex"])}
    for i in range(1, 101)
]

MERCHANTS = [
    {"id": i, "name": fake.company(), "category": cat, "country": fake.country()}
    for i, cat in enumerate([
        "Retail", "Food", "Travel", "Entertainment", "Electronics", 
        "Health", "Automotive", "Home", "Services", "Other"
    ])
]

# Fraud patterns
FRAUD_PATTERNS = [
    {
        "name": "unusual_location",
        "description": "Transaction from unusual location for user",
        "probability": 0.02
    },
    {
        "name": "high_amount",
        "description": "Unusually high transaction amount",
        "probability": 0.03
    },
    {
        "name": "velocity",
        "description": "Rapid succession of transactions",
        "probability": 0.02
    },
    {
        "name": "combination",
        "description": "High amount + unusual location",
        "probability": 0.01
    }
]

# Keep track of user transaction history
user_history = {user["id"]: {
    "last_country": user["country"],
    "last_transaction_time": None,
    "avg_amount": random.uniform(50, 200)
} for user in USERS}

def generate_transaction(forced_fraud=False, fraud_type=None):
    """
    Generate a realistic transaction with optional fraud injection
    """
    # Select user and merchant
    user = random.choice(USERS)
    merchant = random.choice(MERCHANTS)
    user_id = user["id"]
    history = user_history[user_id]
    
    # Base transaction
    transaction = {
        "transaction_id": f"TX-{str(uuid.uuid4())[:8].upper()}",
        "user_id": user_id,
        "user_name": user["name"],
        "card_type": user["card_type"],
        "card_number": f"**** **** **** {random.randint(1000, 9999)}",
        "merchant_id": merchant["id"],
        "merchant_name": merchant["name"],
        "merchant_category": merchant["category"],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "amount": round(random.lognormvariate(mu=np.log(history["avg_amount"]), sigma=0.7), 2),
        "currency": "USD",
        "is_fraud": False,
        "fraud_type": None,
        "risk_score": 0.0
    }
    
    # Normal transaction - select user's home country most of the time
    if random.random() < 0.9 and not forced_fraud:
        transaction["country"] = history["last_country"]
    else:
        # Sometimes pick a different country
        other_countries = [c for c in [fake.country() for _ in range(5)] if c != history["last_country"]]
        transaction["country"] = random.choice(other_countries)
    
    # Decide if transaction is fraudulent
    is_fraud = forced_fraud or (random.random() < 0.05)  # 5% natural fraud rate
    
    if is_fraud:
        # Apply specific fraud pattern if requested
        if fraud_type or (not fraud_type and random.random() < 0.7):
            pattern = next((p for p in FRAUD_PATTERNS if p["name"] == fraud_type), random.choice(FRAUD_PATTERNS))
            
            if pattern["name"] == "unusual_location" or pattern["name"] == "combination":
                # Pick a suspicious country
                suspicious_countries = ["Russia", "Nigeria", "North Korea", "Unknown"]
                transaction["country"] = random.choice(suspicious_countries)
            
            if pattern["name"] == "high_amount" or pattern["name"] == "combination":
                # Make amount suspiciously high
                transaction["amount"] = round(history["avg_amount"] * random.uniform(5, 20), 2)
            
            if pattern["name"] == "velocity":
                # Ensure timestamp is very close to last transaction
                if history["last_transaction_time"]:
                    last_time = datetime.datetime.strptime(history["last_transaction_time"], "%Y-%m-%d %H:%M:%S")
                    new_time = last_time + datetime.timedelta(seconds=random.randint(5, 120))
                    transaction["timestamp"] = new_time.strftime("%Y-%m-%d %H:%M:%S")
            
            transaction["is_fraud"] = True
            transaction["fraud_type"] = pattern["name"]
            transaction["fraud_description"] = pattern["description"]
            transaction["risk_score"] = random.uniform(0.75, 0.99)
        else:
            # Generic fraud with elevated risk score
            transaction["is_fraud"] = True
            transaction["fraud_type"] = "generic"
            transaction["fraud_description"] = "Suspicious activity pattern"
            transaction["risk_score"] = random.uniform(0.65, 0.90)
    else:
        transaction["risk_score"] = random.uniform(0.01, 0.20)
    
    # Update user history
    user_history[user_id]["last_country"] = transaction["country"]
    user_history[user_id]["last_transaction_time"] = transaction["timestamp"]
    user_history[user_id]["avg_amount"] = 0.7 * history["avg_amount"] + 0.3 * transaction["amount"]
    
    return transaction

def generate_transaction_stream(rate=1.0, duration=None, fraud_probability=0.05):
    """
    Generate a continuous stream of transactions
    
    Args:
        rate: Average transactions per second
        duration: How long to generate transactions (None = indefinitely)
        fraud_probability: Chance of injecting a fraud transaction
    """
    start_time = time.time()
    transactions_generated = 0
    
    while True:
        # Check if we should stop
        if duration and (time.time() - start_time) > duration:
            break
        
        # Introduce artificial delay to control transaction rate
        time.sleep(1 / rate)
        
        # Decide if this should be a fraud transaction
        forced_fraud = random.random() < fraud_probability
        fraud_type = None
        if forced_fraud:
            fraud_type = random.choice([p["name"] for p in FRAUD_PATTERNS])
        
        # Generate the transaction
        transaction = generate_transaction(forced_fraud=forced_fraud, fraud_type=fraud_type)
        
        transactions_generated += 1
        yield transaction

if __name__ == "__main__":
    # Example usage
    for i, transaction in enumerate(generate_transaction_stream(rate=2)):
        print(json.dumps(transaction, indent=2))
        if i >= 10:  # Generate 10 transactions then stop
            break