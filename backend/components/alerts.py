import streamlit as st
import uuid
import random
from datetime import datetime

def render_alerts():
    """
    Render the fraud alerts section
    """
    st.subheader(f"Fraud Alerts ({st.session_state.alert_count})")
    st.caption("High-priority notifications")
    
    # Display alerts
    if st.session_state.alerts:
        for alert in st.session_state.alerts:
            severity_color = "red" if alert["severity"] == "high" else "orange"
            with st.expander(f"🚨 {alert['title']} - {alert['txId']}", expanded=True):
                st.markdown(f"""
                **Transaction ID:** {alert['txId']}  
                **Amount:** ${alert['amount']:.2f}  
                **Location:** {alert['location']} {"(unusual for this customer)" if "RU" in alert['location'] or "NG" in alert['location'] else ""}  
                **Device:** {alert['device']}  
                **Why flagged:** {alert['reason']}  
                """)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Block", key=f"block_{alert['id']}"):
                        handle_block_transaction(alert)
                with col2:
                    if st.button("Review", key=f"review_{alert['id']}"):
                        handle_review_transaction(alert)
                with col3:
                    if st.button("Dismiss", key=f"dismiss_{alert['id']}"):
                        handle_dismiss_alert(alert['id'])
    else:
        st.info("No current alerts")

def add_alert(transaction):
    """
    Add a new fraud alert based on transaction data
    
    Args:
        transaction: Dictionary containing transaction data
    """
    # Determine the reason for flagging
    reasons = []
    
    # Check location
    if "RU" in transaction["location"] or "NG" in transaction["location"]:
        reasons.append("Unusual location")
    
    # Check amount
    if transaction["amount"] > 1000:
        reasons.append("High amount")
    
    # Add device info - simulated
    devices = ["New Android device", "iPhone", "Windows PC", "MacBook", "Linux PC", "iPad"]
    device = random.choice(devices)
    
    if "New" in device:
        reasons.append("New device")
    
    if not reasons:
        reasons.append("Suspicious pattern")
    
    # Create the alert
    new_alert = {
        "id": f"AL-{random.randint(100, 999)}",
        "title": "High-Risk Transaction Detected",
        "txId": transaction["id"],
        "amount": transaction["amount"],
        "location": transaction["location"],
        "device": device,
        "reason": " + ".join(reasons),
        "timeDetected": "Just now",
        "severity": "high" if transaction["risk"] > 85 else "medium"
    }
    
    # Add to alerts list
    st.session_state.alerts = [new_alert] + st.session_state.alerts
    st.session_state.alert_count += 1
    
    return new_alert

def handle_block_transaction(alert):
    """
    Handle blocking a flagged transaction
    
    Args:
        alert: The alert dictionary
    """
    # Update transaction status in the list
    for tx in st.session_state.transactions:
        if tx["id"] == alert["txId"]:
            tx["status"] = "Blocked"
    
    # Remove the alert
    handle_dismiss_alert(alert['id'])
    
    # Display confirmation
    st.warning("Transaction blocked and customer notified!")

def handle_review_transaction(alert):
    """
    Handle marking a transaction for manual review
    
    Args:
        alert: The alert dictionary
    """
    # Update transaction status in the list
    for tx in st.session_state.transactions:
        if tx["id"] == alert["txId"]:
            tx["status"] = "Reviewing"
    
    # Update the alert
    for a in st.session_state.alerts:
        if a["id"] == alert["id"]:
            a["title"] = "Under Manual Review"
    
    # Display confirmation
    st.info("Transaction marked for manual review")

def handle_dismiss_alert(alert_id):
    """
    Handle dismissing an alert
    
    Args:
        alert_id: The ID of the alert to dismiss
    """
    # Remove the alert from the list
    st.session_state.alerts = [a for a in st.session_state.alerts if a['id'] != alert_id]
    st.session_state.alert_count = len(st.session_state.alerts)
    
    # Trigger a rerun to update the UI
    st.experimental_rerun()

def get_alert_summary():
    """
    Calculate summary statistics for alerts
    
    Returns:
        Dictionary of alert statistics
    """
    alerts = st.session_state.alerts
    
    # Count by severity
    severity_counts = {
        "high": sum(1 for a in alerts if a["severity"] == "high"),
        "medium": sum(1 for a in alerts if a["severity"] == "medium"),
        "low": sum(1 for a in alerts if a["severity"] == "low")
    }
    
    # Calculate total monetary risk
    total_at_risk = sum(a["amount"] for a in alerts)
    
    return {
        "total_alerts": len(alerts),
        "severity_counts": severity_counts,
        "total_at_risk": total_at_risk
    }