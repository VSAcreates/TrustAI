import pandas as pd
import numpy as np
import uuid

def generate_dataset(num_users=20, transactions_per_user=5):
    np.random.seed(42)
    
    # Generate unique users with balances
    users = []
    for _ in range(num_users):
        users.append({
            "User_ID": str(uuid.uuid4())[:8],  # 8-character unique ID
            "Name": f"User_{np.random.randint(1000,9999)}",
            "Base_Location": np.random.choice(["USA", "UK", "Canada", "India"]),
            "Initial_Balance": np.round(np.random.uniform(1000, 50000), 2)
        })
    
    # Generate transactions
    transactions = []
    for user in users:
        for _ in range(transactions_per_user):
            is_fraud = np.random.choice([0, 1], p=[0.9, 0.1])
            
            if is_fraud:
                amount = np.round(np.random.uniform(3000, 10000), 2)
                location = np.random.choice(["Nigeria", "Russia"])
                device = np.random.choice(["Mobile"])
                behavior = "Abnormal"
            else:
                amount = np.round(np.random.uniform(10, 3000), 2)
                location = user["Base_Location"]
                device = np.random.choice(["Mobile", "Desktop"])
                behavior = "Normal"
            
            transactions.append({
                "Transaction_ID": str(uuid.uuid4())[:12],
                "User_ID": user["User_ID"],
                "Amount": amount,
                "Location": location,
                "Device": device,
                "Behavior": behavior,
                "Timestamp": pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 30)),
                "Is_Fraud": is_fraud
            })
    
    # Create DataFrames
    users_df = pd.DataFrame(users)
    transactions_df = pd.DataFrame(transactions)
    
    # Save to CSV
    users_df.to_csv("user_profiles.csv", index=False)
    transactions_df.to_csv("transaction_history.csv", index=False)
    
    print(f"Generated {num_users} users with {transactions_per_user} transactions each")
    return users_df, transactions_df

if __name__ == "__main__":
    generate_dataset()