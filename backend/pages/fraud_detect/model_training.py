import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def preprocess_data(df):
    df["High_Amount"] = np.where(df["Amount"] > 3000, 1, 0)
    df["Suspicious_Location"] = np.where(df["Location"].isin(["Nigeria", "Russia"]), 1, 0)
    df["Unusual_Behavior"] = np.where(df["Behavior"] == "Abnormal", 1, 0)
    df = pd.get_dummies(df, columns=["Device"])
    return df

def train_model():
    df = pd.read_csv("identity_theft_dataset.csv")
    df_processed = preprocess_data(df)
    
    X = df_processed[["High_Amount", "Suspicious_Location", "Unusual_Behavior", "Device_Mobile"]]
    y = df_processed["Identity_Theft"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, "fraud_detection_model.pkl")
    print("Model saved as fraud_detection_model.pkl")
    return model

if __name__ == "__main__":
    train_model()