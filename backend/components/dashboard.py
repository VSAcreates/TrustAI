import streamlit as st
import time
import random
from datetime import datetime

def render_dashboard():
    """
    Renders the main dashboard layout with header and control bar
    """
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
            from core.transaction_simulator import simulate_transaction
            simulate_transaction()

    with col3:
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
            index=2
        )
        st.session_state.time_range = time_range

    with col4:
        auto_refresh = st.checkbox("Auto Transactions", value=st.session_state.auto_refresh)
        if auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = auto_refresh
            st.experimental_rerun()
    
    return st.session_state.time_range

def render_left_column():
    """
    Renders the left column with account and system information
    """
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
    
    # Account Information Card
    st.subheader("Account Information")
    st.caption("Current account details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Account Holder")
        st.text("John Alen")
    with col2:
        st.caption("Account Number")
        st.text("165489•••••9465")
    
    st.caption("Current Balance")
    st.markdown("<span style='color: #22c55e; font-size: 24px; font-weight: bold;'>$85,000.00</span>", unsafe_allow_html=True)

def initialize_session_state():
    """
    Initialize session state variables if they don't exist
    """
    defaults = {
        "transactions": [
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
                "status": "Reviewing",
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
        ],
        "alerts": [
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
            }
        ],
        "risk_threshold": 70,
        "current_risk": 72,
        "transaction_count": 0,
        "alert_count": 1,
        "is_connected": True,
        "auto_refresh": False,
        "last_model_update": "2 hours ago",
        "time_range": "Last 24 Hours"
    }
    
    # Initialize each variable if it doesn't exist
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def handle_auto_refresh():
    """
    Handle auto-refresh functionality for simulating transactions
    """
    if st.session_state.auto_refresh and st.session_state.is_connected:
        time.sleep(2)  # Pause for 2 seconds
        if random.random() < 0.3:  # 30% chance of generating a transaction
            from core.transaction_simulator import simulate_transaction
            simulate_transaction()
        st.experimental_rerun()