import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_dataset(num_records=1000, output_file="identity_theft_dataset.csv"):
    """
    Generate a synthetic dataset for identity theft detection with realistic patterns.
    
    Parameters:
    -----------
    num_records : int
        Number of records to generate
    output_file : str
        Path to save the generated dataset
        
    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    np.random.seed(42)
    
    # Generate base data
    data = {
        "Transaction_ID": [f"T{10000 + i}" for i in range(num_records)],
        "User_ID": [f"U{np.random.randint(1001, 1101)}" for _ in range(num_records)],
        "Amount": np.round(np.random.uniform(10, 10000, num_records), 2),
        "Location": np.random.choice(
            ["USA", "UK", "Canada", "India", "Nigeria", "Russia", "China", "Brazil", "Germany", "France"],
            num_records,
            p=[0.3, 0.15, 0.1, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05]
        ),
        "Device": np.random.choice(
            ["Mobile", "Desktop", "Tablet"],
            num_records,
            p=[0.6, 0.3, 0.1]
        ),
        "Browser": np.random.choice(
            ["Chrome", "Firefox", "Safari", "Edge", "Opera"],
            num_records,
            p=[0.5, 0.2, 0.15, 0.1, 0.05]
        ),
        "OS": np.random.choice(
            ["Windows", "MacOS", "iOS", "Android", "Linux"],
            num_records,
            p=[0.4, 0.2, 0.2, 0.15, 0.05]
        ),
        "IP_Address": [f"{np.random.randint(1, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}" for _ in range(num_records)],
        "Balance": np.round(np.random.uniform(100, 50000, num_records), 2),
        "Transaction_Time": [datetime.now() - timedelta(hours=np.random.randint(0, 24)) for _ in range(num_records)],
        "Previous_Transactions": np.random.randint(1, 100, num_records),
        "Account_Age_Days": np.random.randint(1, 3650, num_records),  # Up to 10 years
        "Email_Domain": np.random.choice(
            ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "company.com", "suspicious.com"],
            num_records,
            p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
        )
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate behavioral features
    df["Behavior"] = np.where(
        (df["Previous_Transactions"] < 5) & (df["Amount"] > 1000),
        "Abnormal",
        np.random.choice(["Normal", "Abnormal"], num_records, p=[0.85, 0.15])
    )
    
    # Generate time-based features
    df["Hour"] = df["Transaction_Time"].dt.hour
    df["Is_Night"] = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)
    df["Is_Weekend"] = (df["Transaction_Time"].dt.dayofweek >= 5).astype(int)
    
    # Generate device-browser compatibility flags
    df["Device_Browser_Match"] = (
        ((df["Device"] == "Mobile") & (df["Browser"].isin(["Chrome", "Safari"]))) |
        ((df["Device"] == "Desktop") & (df["Browser"].isin(["Chrome", "Firefox", "Edge"]))) |
        ((df["Device"] == "Tablet") & (df["Browser"].isin(["Chrome", "Safari"])))
    ).astype(int)
    
    # Generate target variable based on multiple rules
    df["Identity_Theft"] = np.where(
        (
            # Location-based rules
            (df["Location"].isin(["Nigeria", "Russia", "suspicious.com"])) |
            # Amount-based rules
            ((df["Amount"] > 3000) & (df["Previous_Transactions"] < 5)) |
            # Behavioral rules
            ((df["Behavior"] == "Abnormal") & (df["Amount"] > 1000)) |
            # Time-based rules
            (df["Is_Night"] & (df["Amount"] > 2000)) |
            # Device-based rules
            ((df["Device_Browser_Match"] == 0) & (df["Amount"] > 1500)) |
            # Account age rules
            ((df["Account_Age_Days"] < 30) & (df["Amount"] > 2000))
        ),
        "Yes", "No"
    )
    
    # Add some noise to make it less deterministic
    noise_mask = np.random.random(num_records) < 0.05  # 5% noise
    df.loc[noise_mask, "Identity_Theft"] = np.where(
        df.loc[noise_mask, "Identity_Theft"] == "Yes", "No", "Yes"
    )
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"Dataset generated with {num_records} records")
    print(f"Saved to: {output_file}")
    print(f"\nDataset Summary:")
    print(f"Total records: {len(df)}")
    print(f"Fraudulent transactions: {df['Identity_Theft'].value_counts()['Yes']}")
    print(f"Legitimate transactions: {df['Identity_Theft'].value_counts()['No']}")
    print(f"\nFraud Rate: {(df['Identity_Theft'] == 'Yes').mean():.2%}")
    
    return df

if __name__ == "__main__":
    generate_dataset(1000)  # Generate 1000 records by default 