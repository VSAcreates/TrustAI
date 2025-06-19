import pandas as pd
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz

def create_time_features(df):
    """Create features based on transaction time"""
    if 'TransactionDT' in df.columns:
        START_DATE = '2017-12-01'
        start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
        df['transaction_date'] = df['TransactionDT'].apply(
            lambda x: (start_date + pd.Timedelta(seconds=x)))
        
        df['hour'] = df['transaction_date'].dt.hour
        df['day'] = df['transaction_date'].dt.day
        df['weekday'] = df['transaction_date'].dt.weekday
        df['month'] = df['transaction_date'].dt.month
        
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
    
    return df

def create_amount_features(df):
    """Create features based on transaction amount"""
    if 'TransactionAmt' in df.columns:
        df['rounded_amt_10'] = (df['TransactionAmt'] / 10).round() * 10
        df['rounded_amt_50'] = (df['TransactionAmt'] / 50).round() * 50
        df['rounded_amt_100'] = (df['TransactionAmt'] / 100).round() * 100
        
        df['is_round_amount'] = ((df['TransactionAmt'] % 10) == 0).astype(int)
        
        high_amount_threshold = df['TransactionAmt'].quantile(0.95)
        df['is_high_amount'] = (df['TransactionAmt'] > high_amount_threshold).astype(int)
        
        cents = (df['TransactionAmt'] * 100) % 100
        common_cents = [0, 50, 99, 95, 98]
        df['has_unusual_cents'] = (~cents.isin(common_cents)).astype(int)
    
    return df

def create_card_features(df):
    """Create features based on card information"""
    card_cols = [col for col in df.columns if col.startswith('card')]
    if len(card_cols) >= 2 and 'card1' in df.columns:
        card_counts = df.groupby('card1')[card_cols[1:]].nunique().reset_index()
        card_counts['total_cards'] = card_counts[card_cols[1:]].sum(axis=1)
        
        df = df.merge(card_counts[['card1', 'total_cards']], on='card1', how='left')
        df['has_multiple_cards'] = (df['total_cards'] > 1).astype(int)
    
    return df

def create_email_features(df):
    """Create features based on email information"""
    if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
        df['email_domain_mismatch'] = (df['P_emaildomain'] != df['R_emaildomain']).astype(int)
        
        free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'outlook.com']
        df['is_free_email'] = df['P_emaildomain'].isin(free_domains).astype(int)
        
        suspicious_tlds = ['.xyz', '.top', '.work', '.site']
        df['has_suspicious_tld'] = df['P_emaildomain'].apply(
            lambda x: any(tld in str(x) for tld in suspicious_tlds)
        ).astype(int)
    
    return df

def create_behavioral_features(df):
    """Create features based on behavioral patterns"""
    v_cols = [col for col in df.columns if col.startswith('V')]
    
    if len(v_cols) > 0:
        df['v_missing_count'] = df[v_cols].isna().sum(axis=1)
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
    
    return df

def create_identity_theft_features(df):
    """Create features specifically for identity theft detection"""
    if 'addr1' in df.columns and 'addr2' in df.columns:
        df['addr_mismatch'] = (df['addr1'] != df['addr2']).astype(int)
    
    id_cols = [col for col in df.columns if col.startswith('id-')]
    if len(id_cols) > 0:
        for col in id_cols:
            if col in df.columns:
                df[f'{col}_failed'] = (df[col] <= 0).astype(int)
        
        failed_cols = [col for col in df.columns if col.endswith('_failed')]
        if len(failed_cols) > 0:
            df['total_id_failures'] = df[failed_cols].sum(axis=1)
            df['multiple_id_failures'] = (df['total_id_failures'] > 1).astype(int)
    
    if 'DeviceType' in df.columns and 'browser' in df.columns:
        common_pairs = [
            ('desktop', 'Chrome'), ('desktop', 'Firefox'), ('desktop', 'Edge'),
            ('mobile', 'Chrome Mobile'), ('mobile', 'Safari Mobile')
        ]
        df['device_browser_mismatch'] = (~df[['DeviceType', 'browser']].apply(tuple, axis=1).isin(common_pairs)).astype(int)
    
    return df

def engineer_features(df):
    """Apply all feature engineering functions"""
    df = create_time_features(df)
    df = create_amount_features(df)
    df = create_card_features(df)
    df = create_email_features(df)
    df = create_behavioral_features(df)
    df = create_identity_theft_features(df)
    
    return df