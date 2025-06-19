import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta

def render_transactions_table():
    """
    Render the live transactions table
    """
    st.subheader("Live Transactions")
    st.caption("Real-time transaction monitoring")
    
    # Convert transactions to DataFrame for easier display
    df_transactions = pd.DataFrame(st.session_state.transactions)
    
    # Format status for display
    def format_status(val):
        if val == "Flagged":
            return f"🔴 {val}"
        elif val == "Reviewing":
            return f"🟠 {val}"
        else:
            return f"🟢 {val}"
    
    # Apply formatting
    df_styled = df_transactions.copy()
    df_styled["status"] = df_styled["status"].apply(format_status)
    df_styled["amount"] = df_styled["amount"].apply(lambda x: f"${x:.2f}")
    
    # Display the table with conditional formatting
    st.dataframe(
        df_styled,
        column_order=["id", "amount", "time", "location", "risk", "status"],
        column_config={
            "id": st.column_config.TextColumn("Transaction ID"),
            "amount": st.column_config.TextColumn("Amount"),
            "time": st.column_config.TextColumn("Time"),
            "location": st.column_config.TextColumn("Location"),
            "risk": st.column_config.ProgressColumn(
                "Risk",
                format="%d%%",
                min_value=0,
                max_value=100,
            ),
            "status": st.column_config.TextColumn("Status"),
            "isNew": None,  # Hide this column
        },
        use_container_width=True,
        hide_index=True,
    )

def add_transaction(new_transaction):
    """
    Add a new transaction to the session state
    
    Args:
        new_transaction: Dictionary containing transaction data
    """
    # Format the transaction for the UI
    formatted_transaction = {
        "id": new_transaction["id"],
        "amount": new_transaction["amount"],
        "time": "Just now",
        "location": new_transaction["location"],
        "risk": new_transaction["risk"],
        "status": get_status_from_risk(new_transaction["risk"]),
        "isNew": True
    }
    
    # Update existing transactions' time labels
    for tx in st.session_state.transactions:
        if tx["time"] == "Just now":
            tx["time"] = "1 min ago"
        elif tx["time"] == "1 min ago":
            tx["time"] = "2 min ago"
        elif tx["time"] == "2 min ago":
            tx["time"] = "5 min ago"
        elif tx["time"] == "5 min ago":
            tx["time"] = "10 min ago"
        tx["isNew"] = False
    
    # Add new transaction to the beginning, maintain 5 most recent
    st.session_state.transactions = [formatted_transaction] + st.session_state.transactions[:4]
    st.session_state.transaction_count += 1
    
    # Update the current risk score
    from components.risk_meter import update_risk_score
    update_risk_score(new_transaction["risk"])
    
    # Create alert if high risk
    if new_transaction["risk"] > st.session_state.risk_threshold:
        from components.alerts import add_alert
        add_alert(new_transaction)
    
    return formatted_transaction

def get_status_from_risk(risk_score):
    """
    Determine transaction status based on risk score
    
    Args:
        risk_score: The transaction risk score
        
    Returns:
        String status ("Flagged", "Reviewing", or "Approved")
    """
    if risk_score > st.session_state.risk_threshold:
        return "Flagged"
    elif risk_score > 50:
        return "Reviewing"
    else:
        return "Approved"

def get_transaction_stats():
    """
    Calculate transaction statistics based on current data
    
    Returns:
        Dictionary of statistics
    """
    transactions = st.session_state.transactions
    
    # Calculate total amount
    total_amount = sum(tx["amount"] for tx in transactions)
    
    # Calculate average risk
    avg_risk = sum(tx["risk"] for tx in transactions) / len(transactions) if transactions else 0
    
    # Count by status
    status_counts = {
        "Flagged": sum(1 for tx in transactions if tx["status"] == "Flagged"),
        "Reviewing": sum(1 for tx in transactions if tx["status"] == "Reviewing"),
        "Approved": sum(1 for tx in transactions if tx["status"] == "Approved")
    }
    
    return {
        "total_amount": total_amount,
        "avg_risk": avg_risk,
        "status_counts": status_counts,
        "transaction_count": len(transactions)
    }

def filter_transactions_by_time_range(time_range):
    """
    Filter transactions based on selected time range
    
    Args:
        time_range: String time range selection
        
    Returns:
        List of filtered transactions
    """
    # In a real application, this would filter based on actual timestamps
    # For demo purposes, we'll just return all transactions
    return st.session_state.transactions