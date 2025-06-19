import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
from fuzzywuzzy import fuzz

def load_and_merge_data(train_trans_path, train_id_path, test_trans_path, test_id_path):
    """Load and merge transaction and identity data"""
    train_trans = pd.read_csv(train_trans_path)
    train_id = pd.read_csv(train_id_path)
    test_trans = pd.read_csv(test_trans_path)
    test_id = pd.read_csv(test_id_path)
    
    train = train_trans.merge(train_id, on='TransactionID', how='left')
    test = test_trans.merge(test_id, on='TransactionID', how='left')
    
    return train, test

def standardize_column_names(df):
    """Convert between id_01 and id-01 formats"""
    df.columns = [col.replace('id_', 'id-') if col.startswith('id_') else col 
                 for col in df.columns]
    return df

def get_target_column(df):
    """Identify the target column"""
    if 'isFraud' in df.columns:
        return 'isFraud'
    raise KeyError("'isFraud' column not found in training data")

def ensure_binary_target(y):
    """Convert target to binary if needed"""
    unique_values = np.unique(y)
    if len(unique_values) > 2:
        return (y > y.median()).astype(int)
    return y

def create_identity_features(df):
    """Create features specific to identity theft detection"""
    if 'addr1' in df.columns and 'addr2' in df.columns:
        df['addr_mismatch'] = (df['addr1'] != df['addr2']).astype(int)
        df['addr_distance'] = abs(df['addr1'] - df['addr2'])
    
    if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
        df['email_domain_mismatch'] = (df['P_emaildomain'] != df['R_emaildomain']).astype(int)
        df['email_similarity'] = df.apply(
            lambda x: fuzz.ratio(str(x['P_emaildomain']), str(x['R_emaildomain']))/100,
            axis=1
        )
    
    if 'DeviceInfo' in df.columns:
        df['device_unique'] = df.groupby('DeviceInfo')['DeviceInfo'].transform('count')
    
    if 'TransactionDT' in df.columns and 'card1' in df.columns:
        df['transactions_last_hour'] = df.groupby('card1')['TransactionDT'].transform(
            lambda x: x.diff().lt(3600).cumsum()
        )
    
    return df

def handle_missing_values(df):
    """Handle missing values with more sophisticated strategies"""
    id_cols = [col for col in df.columns if col.startswith('id-')]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            df[f'{col}_missing'] = df[col].isna().astype(int)
    
    trans_cols = ['TransactionAmt', 'TransactionDT']
    for col in trans_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df

def preprocess_data(train, test):
    """Main preprocessing function with 20% sampling"""
    # Standardize column names
    train = standardize_column_names(train)
    test = standardize_column_names(test)
    
    # Create identity-specific features
    train = create_identity_features(train)
    test = create_identity_features(test)
    
    # Handle missing values
    train = handle_missing_values(train)
    test = handle_missing_values(test)
    
    # Handle target
    target_col = get_target_column(train)
    y = ensure_binary_target(train[target_col])
    X = train.drop(target_col, axis=1)
    
    # Ensure all expected columns exist
    expected_id_cols = [f'id-{i:02d}' for i in range(1, 39)]
    for col in expected_id_cols + ['TransactionAmt', 'ProductCD']:
        if col not in X.columns:
            X[col] = 0 if col in expected_id_cols else 'missing'
        if col not in test.columns:
            test[col] = 0 if col in expected_id_cols else 'missing'
    
    # Define feature types
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    # Convert categorical columns to strings
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        test[col] = test[col].astype(str)

    # Create preprocessing pipelines
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # Split data (using only 20% of training data)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training on {len(X_train)} samples (20% of dataset)")
    
    # Fit preprocessor and transform
    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    test_preprocessed = preprocessor.transform(test)
    
    return X_train, X_valid, y_train, y_valid, test_preprocessed, preprocessor