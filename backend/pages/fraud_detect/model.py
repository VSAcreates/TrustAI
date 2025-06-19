import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import joblib
import os
import time

def train_model(X_train, y_train, X_val, y_val, model_type='rf'):
    """Train a fraud detection model optimized for 20% data"""
    print(f"Training {model_type} model on reduced dataset...")
    start_time = time.time()
    
    # Get number of samples correctly for both dense and sparse matrices
    n_samples = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
    print(f"Training samples: {n_samples}")
    
    if model_type == 'rf':
        # Parameters optimized for smaller datasets
        base_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gb':
        base_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    base_model.fit(X_train, y_train)
    
    # Calibrate probabilities with fewer folds
    calibrated_model = CalibratedClassifierCV(base_model, cv=3, method='sigmoid')
    calibrated_model.fit(X_val, y_val)
    
    # Evaluate
    val_probs = calibrated_model.predict_proba(X_val)[:, 1]
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Apply optimal threshold
    val_preds = (val_probs >= optimal_threshold).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_val, val_probs)
    avg_precision = average_precision_score(y_val, val_probs)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Training samples: {n_samples}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Validation ROC AUC: {roc_auc:.4f}")
    print(f"Validation Average Precision: {avg_precision:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds))
    
    return calibrated_model, optimal_threshold

def get_feature_importance(model, feature_names):
    """Extract feature importance from the model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'feature_importances_'):
        importances = model.base_estimator.feature_importances_
    else:
        return None
    
    # Create a DataFrame for better visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return feature_importance

def save_model(model, threshold, preprocessor, output_dir='models'):
    """Save model, threshold and preprocessor"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and preprocessor
    joblib.dump(model, os.path.join(output_dir, 'model.pkl'))
    joblib.dump(threshold, os.path.join(output_dir, 'threshold.pkl'))
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))
    
    print(f"Model saved to {output_dir}")

def load_model(model_path, threshold_path, preprocessor_path):
    """Load model, threshold and preprocessor"""
    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return model, threshold, preprocessor

def plot_roc_curve(y_true, y_score):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    return plt

def plot_precision_recall_curve(y_true, y_score, threshold=None):
    """Plot precision-recall curve"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    
    if threshold is not None:
        idx = np.argmin(np.abs(thresholds - threshold))
        plt.plot(recall[idx], precision[idx], 'ro', markersize=8, 
                 label=f'Threshold = {threshold:.2f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    
    return plt