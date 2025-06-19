import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """
    Load the Kaggle IEEE-CIS Fraud Detection dataset and perform initial exploration
    """
    print("Loading transaction and identity data...")
    
    # Load the data
    train_transaction = pd.read_csv('train_transaction.csv')
    train_identity = pd.read_csv('train_identity.csv')
    
    # Display basic info
    print(f"Transaction data shape: {train_transaction.shape}")
    print(f"Identity data shape: {train_identity.shape}")
    
    # Merge datasets on TransactionID
    data = train_transaction.merge(train_identity, on='TransactionID', how='left')
    print(f"Merged data shape: {data.shape}")
    
    # Check fraud distribution
    fraud_counts = data['isFraud'].value_counts()
    print("\nFraud distribution:")
    print(fraud_counts)
    print(f"Fraud rate: {fraud_counts[1] / len(data) * 100:.2f}%")
    
    # Check missing values
    missing_values = data.isnull().sum()
    missing_ratio = missing_values / len(data) * 100
    missing_summary = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Ratio (%)': missing_ratio
    }).sort_values('Missing Ratio (%)', ascending=False)
    print("\nTop 10 columns with missing values:")
    print(missing_summary.head(10))
    
    # Visualize fraud distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='isFraud', data=data)
    plt.title('Fraud Distribution')
    plt.savefig('fraud_distribution.png')
    
    return data
def add_advanced_features(data, id_cols=['card1']):
    """
    Add advanced features for fraud detection:
    - Transaction velocity (count per hour)
    - Location hops (distance between consecutive transactions)
    - More detailed amount deviations
    """
    print("Adding advanced features...")
    
    # Ensure we have time information
    if 'TransactionDT' not in data.columns:
        print("Warning: TransactionDT not available for velocity calculation")
        return data
    
    # Make a copy to avoid warnings
    enhanced_data = data.copy()
    
    # Add hour of day from TransactionDT
    enhanced_data['hour'] = np.floor((enhanced_data['TransactionDT'] / 3600) % 24)
    
    # For each identifier (like card1, card2, etc.)
    for id_col in id_cols:
        if id_col not in enhanced_data.columns:
            continue
            
        # Group by ID and sort by time
        enhanced_data = enhanced_data.sort_values([id_col, 'TransactionDT'])
        
        # 1. Transaction velocity (count per hour)
        # Create time windows (1 hour = 3600 seconds)
        enhanced_data['hour_bin'] = np.floor(enhanced_data['TransactionDT'] / 3600)
        
        # Count transactions per ID per hour
        tx_counts = enhanced_data.groupby([id_col, 'hour_bin']).size().reset_index()
        tx_counts.columns = [id_col, 'hour_bin', f'{id_col}_tx_per_hour']
        
        # Merge back to original data
        enhanced_data = enhanced_data.merge(tx_counts, on=[id_col, 'hour_bin'], how='left')
        enhanced_data.drop('hour_bin', axis=1, inplace=True)
        
        # 2. Location hops
        if 'addr1' in enhanced_data.columns and 'addr2' in enhanced_data.columns:
            # Use addr1 and addr2 as location proxies
            enhanced_data['location'] = enhanced_data['addr1'].astype(str) + "_" + enhanced_data['addr2'].astype(str)
            
            # Calculate location changes
            enhanced_data[f'{id_col}_prev_location'] = enhanced_data.groupby(id_col)['location'].shift(1)
            enhanced_data[f'{id_col}_location_changed'] = (enhanced_data['location'] != enhanced_data[f'{id_col}_prev_location']).astype(int)
            
            # Count location changes within last 24 hours
            enhanced_data[f'{id_col}_location_hops_24h'] = enhanced_data.groupby(id_col)[f'{id_col}_location_changed'].rolling(
                window=24, min_periods=1).sum().reset_index(level=0, drop=True)
            
            # Clean up temporary columns
            enhanced_data.drop([f'{id_col}_prev_location', 'location'], axis=1, inplace=True)
        
        # 3. More detailed amount deviations
        if 'TransactionAmt' in enhanced_data.columns:
            # Calculate rolling statistics
            enhanced_data[f'{id_col}_amt_mean_7d'] = enhanced_data.groupby(id_col)['TransactionAmt'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean())
            
            enhanced_data[f'{id_col}_amt_std_7d'] = enhanced_data.groupby(id_col)['TransactionAmt'].transform(
                lambda x: x.rolling(window=7, min_periods=1).std())
            
            # Calculate z-score (how many standard deviations from the mean)
            enhanced_data[f'{id_col}_amt_zscore'] = np.where(
                enhanced_data[f'{id_col}_amt_std_7d'] > 0,
                (enhanced_data['TransactionAmt'] - enhanced_data[f'{id_col}_amt_mean_7d']) / enhanced_data[f'{id_col}_amt_std_7d'],
                0
            )
    
    # Fill NaN values
    for col in enhanced_data.columns:
        if enhanced_data[col].dtype != 'object' and enhanced_data[col].isnull().sum() > 0:
            enhanced_data[col] = enhanced_data[col].fillna(0)
    
    return enhanced_data
def preprocess_data(data, test_size=0.2, random_state=42):
    # In your preprocess_data function:
    data = add_advanced_features(data, id_cols=['card1', 'card2', 'card3', 'card4', 'card5', 'card6'])
    """
    Preprocess the data for model training
    """
    print("\nPreprocessing data...")
    
    # Drop high cardinality/high missing value columns and identifiers
    # Adjust this based on your exploration findings
    cols_to_drop = ['TransactionID', 'TransactionDT']
    
    # Find and drop columns with too many missing values (e.g., >75%)
    missing_ratio = data.isnull().sum() / len(data) * 100
    high_missing_cols = missing_ratio[missing_ratio > 75].index.tolist()
    print(f"Dropping {len(high_missing_cols)} columns with >75% missing values")
    
    # Identify card-specific features
    card_cols = [col for col in data.columns if 'card' in col.lower()]
    
    # Prepare feature sets
    categorical_features = []
    numeric_features = []
    
    for col in data.columns:
        if col in cols_to_drop + high_missing_cols + ['isFraud']:
            continue
        
        if data[col].dtype == 'object' or data[col].nunique() < 20:
            categorical_features.append(col)
        else:
            numeric_features.append(col)
    
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numeric features: {len(numeric_features)}")
    
    # Feature engineering
    # Transaction hour from TransactionDT (if you decide to keep it)
    if 'TransactionDT' in data.columns and 'TransactionDT' not in cols_to_drop:
        data['Hour'] = np.floor((data['TransactionDT'] / 3600) % 24)
        numeric_features.append('Hour')
    
    # Amount statistics by card - these can be very predictive
    if 'card1' in data.columns and len(card_cols) > 0:
        # Group amount statistics by card1
        card_amount_stats = data.groupby('card1')['TransactionAmt'].agg(['mean', 'std', 'max']).reset_index()
        card_amount_stats.columns = ['card1', 'card_amt_mean', 'card_amt_std', 'card_amt_max']
        
        # Merge back to dataset
        data = data.merge(card_amount_stats, on='card1', how='left')
        
        # Add to numeric features
        numeric_features.extend(['card_amt_mean', 'card_amt_std', 'card_amt_max'])
    
    # Define preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Prepare features and target
    X = data.drop(['isFraud'] + cols_to_drop + high_missing_cols, axis=1)
    y = data['isFraud']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Fit and transform the data
    print("Fitting preprocessor...")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    print(f"Preprocessed training set shape: {X_train_preprocessed.shape}")
    
    # Save a small sample for demo purposes
    sample_indices = np.random.choice(len(X_test), min(1000, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_indices].copy()
    y_sample = y_test.iloc[sample_indices].copy()
    
    print(f"Demo sample shape: {X_sample.shape}")
    
    # Save the preprocessor for later use with new data
    import joblib
    joblib.dump(preprocessor, 'fraud_preprocessor.joblib')
    
    return (X_train_preprocessed, X_test_preprocessed, y_train, y_test, 
            X_sample, y_sample, preprocessor, X_train.columns.tolist())

if __name__ == "__main__":
    # Execute the preprocessing pipeline
    data = load_and_explore_data()
    X_train, X_test, y_train, y_test, X_sample, y_sample, preprocessor, feature_names = preprocess_data(data)
    
    print("Preprocessing complete! Results saved.")