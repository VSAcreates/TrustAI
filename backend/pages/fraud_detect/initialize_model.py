import os
from utils.preprocessing import load_and_merge_data, preprocess_data
from model import train_model, save_model

def initialize():
    print("Initializing fraud detection model (using 20% of data)...")
    
    # Create directories if needed
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Define paths
    train_trans_path = "data/train_transaction.csv"
    train_id_path = "data/train_identity.csv"
    test_trans_path = "data/test_transaction.csv"
    test_id_path = "data/test_identity.csv"
    
    # Check if files exist
    required_files = [train_trans_path, train_id_path, test_trans_path, test_id_path]
    if not all(os.path.exists(p) for p in required_files):
        print("Error: Missing data files. Please ensure these files exist in data/ folder:")
        print("- train_transaction.csv")
        print("- train_identity.csv")
        print("- test_transaction.csv")
        print("- test_identity.csv")
        return
    
    # Load data
    print("Loading data...")
    train, test = load_and_merge_data(train_trans_path, train_id_path, test_trans_path, test_id_path)
    
    # Sample 20% of training data
    train = train.sample(frac=0.2, random_state=42)
    print(f"Using {len(train)} samples (20% of full dataset)")
    
    # Preprocess
    print("Preprocessing data...")
    X_train, X_val, y_train, y_val, _, preprocessor = preprocess_data(train, test)
    
    # Train model
    print("Training model...")
    model, threshold = train_model(X_train, y_train, X_val, y_val)
    
    # Save model
    print("Saving model...")
    save_model(model, threshold, preprocessor)
    print("Model initialized successfully with 20% of data!")

if __name__ == "__main__":
    initialize()