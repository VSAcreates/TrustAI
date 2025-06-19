import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Configure the Streamlit page
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# # Hide the sidebar completely
# st.markdown("""
#     <style>
#         section[data-testid="stSidebar"] {
#             display: none;
#         }
#     </style>
# """, unsafe_allow_html=True)


# Initialize session state variables if they don't exist
if "transactions" not in st.session_state:
    st.session_state.transactions = [
        {
            "id": "TX-78901",
            "amount": 1299.00,
            "time": "Just now",
            "location": "Moscow, RU",
            "risk": 92,
            "status": "Flagged",
            "isNew": True
        },
        {
            "id": "TX-78900",
            "amount": 499.99,
            "time": "2 min ago",
            "location": "Lagos, NG",
            "risk": 68,
            "status": "Flagged",
            "isNew": False
        },
        {
            "id": "TX-78899",
            "amount": 29.99,
            "time": "5 min ago",
            "location": "New York, US",
            "risk": 12,
            "status": "Approved",
            "isNew": False
        },
        {
            "id": "TX-78898",
            "amount": 149.50,
            "time": "12 min ago",
            "location": "London, UK",
            "risk": 8,
            "status": "Approved",
            "isNew": False
        },
        {
            "id": "TX-78897",
            "amount": 75.25,
            "time": "18 min ago",
            "location": "Toronto, CA",
            "risk": 15,
            "status": "Approved",
            "isNew": False
        }
    ]

if "alerts" not in st.session_state:
    st.session_state.alerts = [
        {
            "id": "AL-001",
            "title": "High-Risk Transaction Detected",
            "txId": "TX-78901",
            "amount": 1299.00,
            "location": "Moscow, RU",
            "device": "New Android device",
            "reason": "Unusual location + New device + High amount",
            "timeDetected": "Just now",
            "severity": "high"
        },
        {
            "id": "AL-002",
            "title": "High-Risk Transaction Detected",
            "txId": "TX-78900",
            "amount": 499.99,
            "location": "Lagos, NG",
            "device": "New Android device",
            "reason": "Unusual location + High amount",
            "timeDetected": "2 min ago",
            "severity": "medium"
        }
    ]

if "risk_threshold" not in st.session_state:
    st.session_state.risk_threshold = 70

if "current_risk" not in st.session_state:
    st.session_state.current_risk = 72

if "transaction_count" not in st.session_state:
    st.session_state.transaction_count = 0

if "alert_count" not in st.session_state:
    st.session_state.alert_count = 2

if "is_connected" not in st.session_state:
    st.session_state.is_connected = True

# if "auto_refresh" not in st.session_state:
#     st.session_state.auto_refresh = False

if "last_model_update" not in st.session_state:
    st.session_state.last_model_update = "2 hours ago"

# Add account balance to session state
if "account_balance" not in st.session_state:
    st.session_state.account_balance = 85000.00

# Add counter for approved transactions
if "approved_count" not in st.session_state:
    st.session_state.approved_count = 0

# Add approval sequence tracker - how many approved transactions in a row
if "approved_sequence" not in st.session_state:
    st.session_state.approved_sequence = 0

# Add flagged counter
if "flagged_needed" not in st.session_state:
    st.session_state.flagged_needed = 0

# Data for simulation
locations = [
    {"city": "New York", "country": "US", "risk": "low"},
    {"city": "London", "country": "UK", "risk": "low"},
    {"city": "Paris", "country": "FR", "risk": "low"},
    {"city": "Moscow", "country": "RU", "risk": "high"},
    {"city": "Lagos", "country": "NG", "risk": "high"},
    {"city": "Toronto", "country": "CA", "risk": "low"},
    {"city": "Sydney", "country": "AU", "risk": "low"},
    {"city": "Beijing", "country": "CN", "risk": "medium"},
    {"city": "Mumbai", "country": "IN", "risk": "medium"},
    {"city": "Kyiv", "country": "UA", "risk": "medium"}
]

devices = ["iPhone", "Android", "New Android device", "Windows PC", "MacBook", "Linux PC", "iPad"]
time_stamps = ["Just now", "1 min ago", "2 min ago", "5 min ago", "8 min ago", "10 min ago"]

def update_transaction_status(transaction_id, new_status):
    """Update the status of a transaction by ID"""
    # First try to find the transaction in the visible list
    for i, transaction in enumerate(st.session_state.transactions):
        if transaction["id"] == transaction_id:
            st.session_state.transactions[i]["status"] = new_status
            # If status is "Approved", deduct the amount from account balance and increment approved counter
            if new_status == "Approved":
                st.session_state.account_balance -= transaction["amount"]
                st.session_state.approved_count += 1
                st.session_state.approved_sequence += 1
            return True
    
    # If transaction not found in visible list, check alerts to get the amount
    for alert in st.session_state.alerts:
        if alert["txId"] == transaction_id:
            # If status is "Approved", deduct the amount from account balance and increment approved counter
            if new_status == "Approved":
                st.session_state.account_balance -= alert["amount"]
                st.session_state.approved_count += 1
                st.session_state.approved_sequence += 1
            return True
    
    return False

def simulate_transaction():
    """Simulate a new transaction with random risk score"""
    # Generate transaction data
    location = random.choice(locations)
    device = random.choice(devices)
    amount = random.uniform(10, 500) if random.random() < 0.8 else random.uniform(500, 3000)
    amount = round(amount, 2)
    
    # Check if we should force a high-risk transaction based on approved count
    force_high_risk = False
    
    if st.session_state.flagged_needed > 0:
        force_high_risk = True
        st.session_state.flagged_needed -= 1
    elif st.session_state.approved_sequence >= random.randint(1, 3):
        force_high_risk = True
        st.session_state.approved_sequence = 0
        st.session_state.flagged_needed = random.randint(0, 1)  # May need one more flagged after this
    
    # Calculate risk based on multiple factors
    if force_high_risk:
        # Force high risk to ensure it exceeds threshold
        risk = st.session_state.risk_threshold + random.randint(10, 30)
        
        # If we want a very high risk, use a high-risk location
        if random.random() < 0.7:
            location = next((loc for loc in locations if loc["risk"] == "high"), location)
            
        # Make amount larger for high-risk transactions
        if random.random() < 0.7:
            amount = random.uniform(1000, 3000)
            amount = round(amount, 2)
    else:
        risk = 0
        # Location factor
        if location["risk"] == "high":
            risk += 40
        elif location["risk"] == "medium":
            risk += 20
        
        # Amount factor
        if amount > 1000:
            risk += 30
        elif amount > 500:
            risk += 15
        
        # Random factor (model uncertainty)
        risk += random.randint(0, 30)
    
    # Cap risk at 100
    risk = min(risk, 100)
    
    # Generate transaction ID (increment from last ID)
    last_id = st.session_state.transactions[0]["id"] if st.session_state.transactions else "TX-78896"
    id_num = int(last_id.split("-")[1]) + 1
    new_id = f"TX-{id_num}"
    
    # Only two statuses: Flagged or Approved (removed Reviewing status)
    status = "Flagged" if risk > st.session_state.risk_threshold else "Approved"
    
    # Create new transaction
    new_transaction = {
        "id": new_id,
        "amount": amount,
        "time": "Just now",
        "location": f"{location['city']}, {location['country']}",
        "risk": risk,
        "status": status,
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
    
    # Update transactions (add to beginning, maintain 5 most recent)
    st.session_state.transactions = [new_transaction] + st.session_state.transactions[:4]
    st.session_state.transaction_count += 1
    
    # If status is "Approved", deduct the amount from account balance and increment approved counter
    if status == "Approved":
        st.session_state.account_balance -= amount
        st.session_state.approved_count += 1
        st.session_state.approved_sequence += 1
    
    # Create alert if high risk
    if risk > st.session_state.risk_threshold:
        reasons = []
        if location["risk"] == "high":
            reasons.append("Unusual location")
        if amount > 1000:
            reasons.append("High amount")
        if "New" in device:
            reasons.append("New device")
        
        if not reasons:
            reasons.append("Suspicious pattern")
        
        new_alert = {
            "id": f"AL-{random.randint(100, 999)}",
            "title": "High-Risk Transaction Detected",
            "txId": new_id,
            "amount": amount,
            "location": f"{location['city']}, {location['country']}",
            "device": device,
            "reason": " + ".join(reasons),
            "timeDetected": "Just now",
            "severity": "high" if risk > 85 else "medium"
        }
        
        st.session_state.alerts = [new_alert] + st.session_state.alerts
        st.session_state.alert_count += 1
    
    # Update current risk (moving average)
    st.session_state.current_risk = round((st.session_state.current_risk * 2 + risk) / 3)

# Header
st.title("🛡️ Fraud Detection Dashboard")

# Top control bar
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    connection_status = "Connected" if st.session_state.is_connected else "Disconnected"
    connection_color = "green" if st.session_state.is_connected else "red"
    st.markdown(f"<span style='color:{connection_color}'>●</span> {connection_status}", unsafe_allow_html=True)

with col2:
    if st.button("Simulate Transaction"):
        simulate_transaction()

with col3:
    time_range = st.selectbox(
        "Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
        index=2
    )

# with col4:
#     auto_refresh = st.checkbox("Auto Transactions", value=st.session_state.auto_refresh)
#     st.session_state.auto_refresh = auto_refresh

# Main content in 3 columns
col_left, col_mid_right = st.columns([1, 2])

# Left column - Account information and Risk Meter
with col_left:
    # Account Information Card
    st.subheader("Account Information")
    st.caption("Current account details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Account Holder")
        st.text("Abhsihek Mishra")
    with col2:
        st.caption("Account Number")
        st.text("165489•••••9465")
    
    st.caption("Current Balance")
    st.markdown(f"<span style='color: #22c55e; font-size: 24px; font-weight: bold;'>₹{st.session_state.account_balance:.2f}</span>", unsafe_allow_html=True)
    
    # Risk Meter Card
    st.subheader("Risk Meter")
    st.caption("Current fraud risk level")
    
    # Create risk gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=st.session_state.current_risk,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': 
                   "#ef4444" if st.session_state.current_risk > 80 else 
                   "#f97316" if st.session_state.current_risk > 50 else 
                   "#22c55e"},
            'steps': [
                {'range': [0, 50], 'color': 'rgba(34, 197, 94, 0.2)'},
                {'range': [50, 80], 'color': 'rgba(249, 115, 22, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': st.session_state.risk_threshold
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk threshold slider
    st.caption(f"Risk Threshold: {st.session_state.risk_threshold}%")
    risk_threshold = st.slider("Risk Threshold", 0, 100, st.session_state.risk_threshold, label_visibility="collapsed")
    if risk_threshold != st.session_state.risk_threshold:
        st.session_state.risk_threshold = risk_threshold
    
    # System Stats Card
    st.subheader("System Stats")
    st.caption("Detection system performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Transactions", st.session_state.transaction_count)
    with col2:
        st.metric("Fraud Alerts", st.session_state.alert_count)
        
    st.caption("Last Model Update")
    st.text(st.session_state.last_model_update)

# Middle and right column - Transaction Monitor & Alerts
with col_mid_right:
    # Live Transactions
    st.subheader("Live Transactions")
    st.caption("Real-time transaction monitoring")
    
    # Convert transactions to DataFrame for easier display
    df_transactions = pd.DataFrame(st.session_state.transactions)
    
    # Format the DataFrame
    def highlight_risk(val):
        if val > 80:
            return 'background-color: rgba(239, 68, 68, 0.2)'
        elif val > 50:
            return 'background-color: rgba(249, 115, 22, 0.2)'
        else:
            return ''
    
    def format_status(val):
        if val == "Flagged":
            return f"🔴 {val}"
        elif val == "Blocked":
            return f"⚫ {val}"
        else:
            return f"🟢 {val}"
    
    # Apply formatting
    df_styled = df_transactions.copy()
    df_styled["status"] = df_styled["status"].apply(format_status)
    df_styled["amount"] = df_styled["amount"].apply(lambda x: f"₹{x:.2f}")
    
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
    
    # Fraud Alerts Section
    st.subheader(f"Fraud Alerts ({st.session_state.alert_count})")
    st.caption("High-priority notifications")
    
    # Display alerts
    if st.session_state.alerts:
        for alert in st.session_state.alerts:
            severity_color = "red" if alert["severity"] == "high" else "orange"
            with st.expander(f"🚨 {alert['title']} - {alert['txId']}", expanded=True):
                st.markdown(f"""
                **Transaction ID:** {alert['txId']}  
                **Amount:** ₹{alert['amount']:.2f}  
                **Location:** {alert['location']} {"(unusual for this customer)" if "RU" in alert['location'] or "NG" in alert['location'] else ""}  
                **Device:** {alert['device']}  
                **Why flagged:** {alert['reason']}  
                """)
                
                col1, col2 = st.columns(2)  # Changed from 3 columns to 2 columns (removed Review)
                with col1:
                    if st.button("Block", key=f"block_{alert['id']}"):
                        update_transaction_status(alert['txId'], "Blocked")
                        st.warning("Transaction blocked and customer notified!")
                        # Remove alert after action
                        st.session_state.alerts = [a for a in st.session_state.alerts if a['id'] != alert['id']]
                        st.session_state.alert_count -= 1
                        st.rerun()
                with col2:
                    if st.button("Approve", key=f"dismiss_{alert['id']}"):
                        update_transaction_status(alert['txId'], "Approved")
                        # Remove alert after action
                        st.session_state.alerts = [a for a in st.session_state.alerts if a['id'] != alert['id']]
                        st.session_state.alert_count -= 1
                        st.rerun()
    else:
        st.info("No current alerts")

# # Add new buttons at the bottom
# col1, col2 = st.columns(2)
# with col1:
#     if st.button("Phishing URL Detection", use_container_width=True):
#         st.info("Phishing URL Detection system activated. Scanning for suspicious URLs...")

# with col2:
#     if st.button("FRAUD CALL Detection", use_container_width=True):
#         try:
#             # Option 1: Use the proper page path if you have a multipage app structure
#             st.switch_page("pages/Scam Alert.py")
#         except Exception as e:
#             # Option 2: Fallback - create a session state variable to track navigation
#             st.session_state.show_fraud_call_page = True
#             st.rerun()
# # Add this code outside the column structure to handle the navigation
# if 'show_fraud_call_page' in st.session_state and st.session_state.show_fraud_call_page:
#     # Clear the flag
#     st.session_state.show_fraud_call_page = False
    
#     # Clear the current UI
#     st.empty()
    
#     # Render the fraud call detection UI
#     st.title("🔍 Fraud Call Detection")
#     st.subheader("Monitor and detect suspicious calls")
    
#     # Here you would add your fraud call detection UI components
#     st.info("Fraud call detection system activated. Monitoring for suspicious calls...")
    
#     # Add a back button
#     if st.button("← Back to Dashboard"):
#         st.rerun()

# # Auto-refresh for simulation - changed to 5 seconds and always generate transactions
# if st.session_state.auto_refresh and st.session_state.is_connected:
#     time.sleep(5)  # Pause for 5 seconds (changed from 2)
#     simulate_transaction()  # Always generate a transaction
#     st.experimental_rerun()