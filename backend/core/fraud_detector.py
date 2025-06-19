import pandas as pd
import numpy as np
import datetime
import joblib
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self, model_path='models/fraud_detection_rf_model.joblib', 
                 preprocessor_path='models/fraud_preprocessor.joblib'):
        """
        Initialize the fraud detector with pretrained model and preprocessor
        """
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info("Model and preprocessor loaded successfully")
        except FileNotFoundError:
            logger.warning("Model or preprocessor not found. Using rule-based detection only.")
            self.model = None
            self.preprocessor = None
        
        # For tracking user behavior
        self.user_profiles = defaultdict(lambda: {
            'countries': set(),
            'transactions': [],
            'avg_amount': 0,
            'transaction_count': 0,
            'last_transaction_time': None
        })
        
        # Rules thresholds
        self.thresholds = {
            'amount_multiplier': 5.0,    # Multiple of user's average amount that is suspicious
            'velocity_seconds': 300,      # Time window for velocity check (5 minutes)
            'velocity_threshold': 5,      # Number of transactions in window to trigger alert
            'amount_threshold': 3000.0,   # Absolute amount threshold for high-value transactions
            'risk_threshold': 0.7         # Risk score threshold for ML model
        }

    def process_transaction(self, transaction):
        """
        Process a single transaction and determine if it's fraudulent
        """
        # Extract transaction details
        user_id = transaction['user_id']
        amount = float(transaction['amount'])
        country = transaction['country']
        timestamp = datetime.datetime.strptime(transaction['timestamp'], "%Y-%m-%d %H:%M:%S")
        
        # Get user profile, creating a new one if necessary
        profile = self.user_profiles[user_id]
        
        # Update user profile
        if profile['transaction_count'] == 0:
            profile['avg_amount'] = amount
        else:
            # Weighted average: give more weight to historical average
            profile['avg_amount'] = (0.8 * profile['avg_amount']) + (0.2 * amount)
        
        profile['countries'].add(country)
        profile['transaction_count'] += 1
        
        # Store transaction timestamp for velocity check
        recent_txns = [tx for tx in profile['transactions'] 
                      if (timestamp - tx).total_seconds() < self.thresholds['velocity_seconds']]
        recent_txns.append(timestamp)
        profile['transactions'] = recent_txns
        profile['last_transaction_time'] = timestamp
        
        # Calculate risk factors
        risk_factors = []
        
        # Rule 1: Amount significantly above user average
        amount_ratio = amount / profile['avg_amount'] if profile['avg_amount'] > 0 else 1
        if amount_ratio > self.thresholds['amount_multiplier']:
            risk_factors.append({
                'type': 'high_amount_ratio',
                'description': f'Amount {amount_ratio:.1f}x user average',
                'score': min(0.3 + (amount_ratio / 20), 0.9)
            })
        
        # Rule 2: High absolute amount
        if amount > self.thresholds['amount_threshold']:
            risk_factors.append({
                'type': 'high_absolute_amount',
                'description': f'High value transaction (${amount:.2f})',
                'score': min(0.4 + (amount / 10000), 0.85)
            })
        
        # Rule 3: New country for user
        if len(profile['countries']) > 1 and profile['transaction_count'] > 3:
            risk_factors.append({
                'type': 'unusual_location',
                'description': f'Transaction from {country} (user has used {len(profile["countries"])} countries)',
                'score': 0.65
            })
        
        # Rule 4: Transaction velocity
        if len(recent_txns) > self.thresholds['velocity_threshold']:
            risk_factors.append({
                'type': 'high_velocity',
                'description': f'{len(recent_txns)} transactions in {self.thresholds["velocity_seconds"]/60:.1f} minutes',
                'score': min(0.5 + (len(recent_txns) / 10), 0.95)
            })
            
        # Calculate rule-based risk score (max of all risk factors)
        rule_score = max([factor['score'] for factor in risk_factors]) if risk_factors else 0.0
        
        # ML-based scoring if model is available
        ml_score = 0.0
        if self.model and self.preprocessor:
            try:
                # Prepare transaction for model
                # This is simplified - in practice, adapt your transaction data to match
                # what your model expects based on the preprocessing pipeline
                tx_df = pd.DataFrame({
                    'TransactionAmt': [amount],
                    'card1': [user_id % 10000],  # Simplification
                    'hour': [timestamp.hour],
                    # Add other required fields with sensible defaults
                })
                
                # Apply preprocessing and predict
                X = self.preprocessor.transform(tx_df)
                ml_score = float(self.model.predict_proba(X)[0, 1])
                
                if ml_score > self.thresholds['risk_threshold']:
                    risk_factors.append({
                        'type': 'ml_model',
                        'description': f'ML model detected suspicious pattern',
                        'score': ml_score
                    })
            except Exception as e:
                logger.error(f"Error in ML prediction: {e}")
        
        # Combine rule-based and ML scores
        # Use 60% rule-based, 40% ML if available
        final_score = rule_score if ml_score == 0 else (0.6 * rule_score + 0.4 * ml_score)
        
        # Prepare result
        result = {
            'transaction_id': transaction['transaction_id'],
            'user_id': user_id,
            'amount': amount,
            'risk_score': round(final_score, 2),
            'risk_factors': risk_factors,
            'is_fraud': final_score >= self.thresholds['risk_threshold'],
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Transaction {transaction['transaction_id']} - Risk Score: {final_score:.2f}")
        
        return result

    def update_thresholds(self, **kwargs):
        """
        Update the detection thresholds
        """
        for key, value in kwargs.items():
            if key in self.thresholds:
                self.thresholds[key] = value
                logger.info(f"Updated threshold: {key} = {value}")
        
        return self.thresholds