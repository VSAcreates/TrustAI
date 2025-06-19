import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
import os
import joblib

def load_and_explore_data(sample_fraction=0.1, random_state=42):
    """
    Load the Kaggle IEEE-CIS Fraud Detection dataset and perform initial exploration
    Uses sampling to reduce memory usage
    """
    print(f"Loading transaction and identity data (sampling {sample_fraction*100}% of data)...")
    
    # Load the data with sampling to reduce memory usage
    train_transaction = pd.read_csv('train_transaction.csv')
    
    # Sample data to reduce memory usage
    if sample_fraction < 1.0:
        # Stratified sampling to maintain fraud distribution
        fraud = train_transaction[train_transaction['isFraud'] == 1]
        non_fraud = train_transaction[train_transaction['isFraud'] == 0]
        
        # Keep all fraud cases but sample from non-fraud
        non_fraud_sample = non_fraud.sample(
            frac=sample_fraction, 
            random_state=random_state
        )
        
        train_transaction = pd.concat([fraud, non_fraud_sample])
        print(f"Sampled data: {len(train_transaction)} transactions")
    
    # Load identity data
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
    plt.close()
    
    return data

def reduce_memory_usage(df):
    """
    Reduce memory usage of a dataframe by downcasting numeric columns
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage before optimization: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization: {end_mem:.2f} MB")
    print(f"Memory reduced by {100 * (start_mem - end_mem) / start_mem:.2f}%")
    
    return df

def add_basic_features(data):
    """
    Add simplified features for fraud detection
    Much more memory efficient than the original function
    """
    print("Adding basic features...")
    
    # Make a copy to avoid warnings
    enhanced_data = data.copy()
    
    # Add hour of day from TransactionDT
    if 'TransactionDT' in enhanced_data.columns:
        enhanced_data['hour'] = (enhanced_data['TransactionDT'] / 3600) % 24
        enhanced_data['hour'] = enhanced_data['hour'].astype(np.int8)
    
    # Transaction amount features
    if 'TransactionAmt' in enhanced_data.columns:
        # Round amount to reduce unique values
        enhanced_data['TransactionAmt_log'] = np.log1p(enhanced_data['TransactionAmt']).astype(np.float32)
        
        # Card-based transaction amount stats (memory efficient)
        if 'card1' in enhanced_data.columns:
            card_groups = enhanced_data.groupby('card1')['TransactionAmt']
            enhanced_data['card1_amt_mean'] = enhanced_data['card1'].map(card_groups.mean()).astype(np.float32)
            enhanced_data['card1_amt_std'] = enhanced_data['card1'].map(card_groups.std()).astype(np.float32)
    
    # Email domain feature
    if 'P_emaildomain' in enhanced_data.columns:
        enhanced_data['email_domain_group'] = enhanced_data['P_emaildomain'].fillna('missing')
        common_domains = ['gmail', 'yahoo', 'hotmail', 'outlook', 'aol']
        for domain in common_domains:
            enhanced_data.loc[enhanced_data['email_domain_group'].str.contains(domain, na=False), 'email_domain_group'] = domain
        enhanced_data.loc[~enhanced_data['email_domain_group'].isin(common_domains + ['missing']), 'email_domain_group'] = 'other'
    
    return enhanced_data

def preprocess_data(data, test_size=0.2, random_state=42):
    """
    Preprocess the data for model training with memory optimization
    """
    print("\nPreprocessing data...")
    
    # Add basic features
    data = add_basic_features(data)
    
    # Reduce memory usage
    data = reduce_memory_usage(data)
    
    # Drop high cardinality/high missing value columns and identifiers
    cols_to_drop = ['TransactionID', 'TransactionDT']
    
    # Find and drop columns with too many missing values (e.g., >75%)
    missing_ratio = data.isnull().sum() / len(data) * 100
    high_missing_cols = missing_ratio[missing_ratio > 75].index.tolist()
    print(f"Dropping {len(high_missing_cols)} columns with >75% missing values")
    
    # Use feature selection to reduce dimensionality
    num_features_to_keep = 100  # Adjust this based on your memory constraints
    
    # Define categories based on column names
    id_cols = [col for col in data.columns if col.startswith('id_')]
    card_cols = [col for col in data.columns if col.startswith('card')]
    addr_cols = [col for col in data.columns if col.startswith('addr')]
    
    # Select columns to keep
    columns_to_keep = ['isFraud', 'TransactionAmt', 'TransactionAmt_log', 
                     'card1_amt_mean', 'card1_amt_std', 'hour']
    
    # Add some columns from each category
    for col_list in [id_cols, card_cols, addr_cols]:
        missing_in_category = missing_ratio[col_list].sort_values()
        # Keep the columns with least missing values
        new_cols = missing_in_category.head(min(5, len(missing_in_category))).index.tolist()
        # Only add columns that aren't already in columns_to_keep
        columns_to_keep.extend([col for col in new_cols if col not in columns_to_keep])
    
    # Add other potentially useful columns with low missing values
    remaining_cols = [col for col in data.columns 
                     if col not in columns_to_keep + cols_to_drop + high_missing_cols + ['isFraud']]
    remaining_missing = missing_ratio[remaining_cols].sort_values()
    new_cols = remaining_missing.head(num_features_to_keep - len(columns_to_keep)).index.tolist()
    # Only add columns that aren't already in columns_to_keep
    columns_to_keep.extend([col for col in new_cols if col not in columns_to_keep])
    
    # Keep only selected columns
    selected_data = data[columns_to_keep]
    print(f"Selected {len(columns_to_keep)} features for model training")
    
    # Separate features and target
    X = selected_data.drop('isFraud', axis=1)
    y = selected_data['isFraud']
    
    # Identify categorical and numerical columns
    categorical_features = []
    for col in X.columns:
        n_unique = X[col].nunique()
        if pd.api.types.is_object_dtype(X[col]) or (isinstance(n_unique, (int, np.integer)) and n_unique < 20):
            categorical_features.append(col)
    
    numeric_features = [col for col in X.columns if col not in categorical_features]
    
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numeric features: {len(numeric_features)}")
    
    # Define preprocessing for numeric features - use float32 instead of float64
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features - use limited categories
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=10))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any other columns not specified
    )
    
    # Split the data - memory efficient way
    from sklearn.model_selection import train_test_split
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
    joblib.dump(preprocessor, 'fraud_preprocessor.joblib')
    
    return (X_train_preprocessed, X_test_preprocessed, y_train, y_test, 
            X_sample, y_sample, preprocessor, X_train.columns.tolist())

def train_random_forest(X_train, y_train, class_weight='balanced', random_state=42):
    """
    Train a Random Forest model with efficient parameter settings
    
    Args:
        X_train: Preprocessed training features
        y_train: Training target variable
        class_weight: Handling class imbalance
        random_state: For reproducibility
    
    Returns:
        Trained Random Forest model
    """
    print("\nTraining Random Forest model...")
    
    # Create memory-efficient RF with limited parameters
    rf = RandomForestClassifier(
        n_estimators=100,  # Reduced from potentially 200
        max_depth=20,      # Limited to prevent memory issues
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    # Use cross-validation for evaluation without full grid search
    # This is much more memory efficient
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='recall')
    
    print(f"Cross-validation recall scores: {cv_scores}")
    print(f"Mean CV recall: {cv_scores.mean():.4f}")
    
    # Now fit the model on the full training data
    print("Fitting final model on full training data...")
    rf.fit(X_train, y_train)
    
    return rf

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Evaluate the model performance with various metrics and visualizations
    """
    print("\nEvaluating model performance...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Feature importance if we have the feature names
    if feature_names is not None and hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, feature_names)
    
    # Finding optimal threshold for fraud detection
    find_optimal_threshold(y_test, y_pred_proba)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm
    }

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance from the model
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to show
    """
    # Check if we have the right number of feature names
    if len(feature_names) != model.feature_importances_.shape[0]:
        print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match feature importances shape ({model.feature_importances_.shape[0]})")
        feature_names = [f"Feature_{i}" for i in range(model.feature_importances_.shape[0])]
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    return feature_importance

def find_optimal_threshold(y_true, y_score):
    """
    Find optimal threshold for classification based on precision-recall trade-off
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    # Calculate F1 score for different thresholds
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    
    # Find threshold with best F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
    
    # Calculate scores at different thresholds
    thresholds_to_check = [0.1, 0.2, 0.3, 0.4, 0.5, optimal_threshold, 0.7, 0.8, 0.9]
    threshold_results = []
    
    for threshold in thresholds_to_check:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        threshold_results.append({
            'Threshold': threshold,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn
        })
    
    # Create DataFrame for better visualization
    threshold_df = pd.DataFrame(threshold_results)
    
    print("\nPerformance at different thresholds:")
    print(threshold_df.round(3))
    
    print(f"\nOptimal threshold (max F1): {optimal_threshold:.3f}")
    
    return optimal_threshold

def predict_new_transactions(transactions_df, model_path='fraud_detection_rf_model.joblib', 
                          preprocessor_path='fraud_preprocessor.joblib'):
    """
    Make predictions on new transaction data
    
    Args:
        transactions_df: DataFrame with new transactions
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor
    
    Returns:
        DataFrame with predictions and risk scores
    """
    # Load model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Preprocess new data
    X_new = preprocessor.transform(transactions_df)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_new)[:, 1]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'TransactionID': transactions_df.index,
        'fraud_probability': y_pred_proba,
        'fraud_prediction': (y_pred_proba >= 0.5).astype(int)
    })
    
    # Add risk category
    results['risk_category'] = pd.cut(
        results['fraud_probability'],
        bins=[0, 0.3, 0.7, 1],
        labels=['Low', 'Medium', 'High']
    )
    
    return results

def main():
    # Step 1: Load and preprocess data with sampling to reduce memory usage
    print("Starting memory-optimized fraud detection model training...")
    data = load_and_explore_data(sample_fraction=0.2)  # Use 20% of data
    
    # Step 2: Preprocess the data and extract features
    X_train, X_test, y_train, y_test, X_sample, y_sample, preprocessor, feature_names = preprocess_data(data)
    
    # Step 3: Train the Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    
    # Step 4: Evaluate the model
    results = evaluate_model(rf_model, X_test, y_test, feature_names)
    
    # Step 5: Save the model
    joblib.dump(rf_model, 'fraud_detection_rf_model.joblib')
    print("\nModel saved as 'fraud_detection_rf_model.joblib'")
    
    return rf_model, preprocessor

if __name__ == "__main__":
    rf_model, preprocessor = main()