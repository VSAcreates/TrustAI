import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta
import uuid

def format_currency(amount):
    """
    Format a number as currency
    
    Args:
        amount: The numeric amount to format
        
    Returns:
        Formatted currency string
    """
    return f"${amount:,.2f}"

def format_percentage(value):
    """
    Format a number as percentage
    
    Args:
        value: The numeric value to format
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.1f}%"

def format_timestamp(timestamp):
    """
    Format a timestamp into a human-readable string
    
    Args:
        timestamp: Datetime object or string timestamp
        
    Returns:
        Human-readable time string
    """
    if isinstance(timestamp, str):
        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    
    now = datetime.now()
    diff = now - timestamp
    
    if diff.total_seconds() < 60:
        return "Just now"
    elif diff.total_seconds() < 3600:
        minutes = int(diff.total_seconds() / 60)
        return f"{minutes} min ago"
    elif diff.total_seconds() < 86400:
        hours = int(diff.total_seconds() / 3600)
        return f"{hours} hours ago"
    else:
        days = int(diff.total_seconds() / 86400)
        return f"{days} days ago"

def generate_transaction_id():
    """
    Generate a unique transaction ID
    
    Returns:
        String transaction ID
    """
    return f"TX-{str(uuid.uuid4())[:8].upper()}"

def generate_alert_id():
    """
    Generate a unique alert ID
    
    Returns:
        String alert ID
    """
    return f"AL-{random.randint(100, 999)}"

def calculate_risk_score(transaction):
    """
    Calculate risk score based on transaction details
    
    Args:
        transaction: Dictionary containing transaction data
        
    Returns:
        Risk score (0-100)
    """
    risk = 0
    
    # Location factor
    high_risk_locations = ["RU", "NG", "CN", "UA"]
    if any(loc in transaction["location"] for loc in high_risk_locations):
        risk += 40
    
    # Amount factor
    if transaction["amount"] > 1000:
        risk += 30
    elif transaction["amount"] > 500:
        risk += 15
    
    # Time factor - transactions at odd hours
    if isinstance(transaction.get("timestamp"), datetime):
        hour = transaction["timestamp"].hour
        if hour < 6 or hour > 22:  # Between 10 PM and 6 AM
            risk += 20
    
    # Random factor (model uncertainty)
    risk += random.randint(0, 20)
    
    # Cap risk at 100
    return min(risk, 100)

def load_sample_data():
    """
    Load sample transaction data for demonstration
    
    Returns:
        DataFrame of sample transactions
    """
    # Create sample data
    data = []
    now = datetime.now()
    
    for i in range(20):
        timestamp = now - timedelta(minutes=random.randint(5, 60))
        
        # Create transaction with somewhat realistic data
        transaction = {
            "id": f"TX-{78896 + i}",
            "timestamp": timestamp,
            "amount": random.uniform(10, 1500),
            "location": random.choice([
                "New York, US", "London, UK", "Paris, FR", 
                "Moscow, RU", "Lagos, NG", "Toronto, CA"
            ]),
            "device": random.choice([
                "iPhone", "Android", "Windows PC", "MacBook", "New Device"
            ]),
            "user_id": random.randint(1000, 9999)
        }
        
        # Calculate risk score
        transaction["risk"] = calculate_risk_score(transaction)
        
        # Determine status based on risk
        if transaction["risk"] > 70:
            transaction["status"] = "Flagged"
        elif transaction["risk"] > 50:
            transaction["status"] = "Reviewing"
        else:
            transaction["status"] = "Approved"
        
        # Format timestamp for display
        transaction["time"] = format_timestamp(timestamp)
        
        data.append(transaction)
    
    return pd.DataFrame(data)

def create_time_series_data(base_value, periods=30, volatility=0.2):
    """
    Create time series data with random walk
    
    Args:
        base_value: Starting value for the series
        periods: Number of periods to generate
        volatility: How much randomness to add
        
    Returns:
        DataFrame with date and value columns
    """
    dates = pd.date_range(end=datetime.now(), periods=periods)
    values = [base_value]
    
    for i in range(1, periods):
        change = base_value * volatility * np.random.randn()
        new_value = max(0, values[-1] + change)
        values.append(new_value)
    
    return pd.DataFrame({
        'date': dates,
        'value': values
    })

def get_color_for_risk(risk_score):
    """
    Get appropriate color based on risk score
    
    Args:
        risk_score: Numeric risk score (0-100)
        
    Returns:
        Hex color code
    """
    if risk_score > 80:
        return "#ef4444"  # Red
    elif risk_score > 50:
        return "#f97316"  # Orange
    else:
        return "#22c55e"  # Green