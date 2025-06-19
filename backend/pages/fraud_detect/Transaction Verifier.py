import streamlit as st
import pandas as pd
import joblib
from pages.fraud_detect.email_alert import EmailAlertSystem
from datetime import datetime
import numpy as np
import base64
import json

# UI Configuration - Must be first Streamlit command
st.set_page_config(
    page_title="Bank Fraud Detection",
    layout="wide",
    page_icon="🏦"
)

# Initialize services
email_system = EmailAlertSystem()

# Load data
@st.cache_data
def load_data():
    users = pd.read_csv("pages/fraud_detect/user_profiles.csv")
    transactions = pd.read_csv("pages/fraud_detect/transaction_history.csv")
    return users, transactions

users_df, transactions_df = load_data()

# Load model
@st.cache_resource
def load_model():
    return joblib.load("pages/fraud_detect/fraud_detection_model.pkl")

model = load_model()

def handle_transaction_action(action, encoded_data):
    """Handle transaction approval or rejection from email link"""
    try:
        # Decode the transaction data
        decoded_data = base64.b64decode(encoded_data).decode()
        txn_data = json.loads(decoded_data)
        
        if action == "approve":
            # Update balances
            users_df.loc[users_df["User_ID"] == txn_data['user_id'], "Initial_Balance"] -= txn_data['amount']
            users_df.loc[users_df["User_ID"] == txn_data['recipient_id'], "Initial_Balance"] += txn_data['amount']
            
            # Update transaction status
            transactions_df.loc[transactions_df['Transaction_ID'] == txn_data['transaction_id'], 'Is_Fraud'] = 0
            
            # Save changes
            transactions_df.to_csv("pages/fraud_detect/transaction_history.csv", index=False)
            users_df.to_csv("pages/fraud_detect/user_profiles.csv", index=False)
            
            st.success("Transaction approved successfully!")
        else:
            # Just update the transaction status to rejected
            transactions_df.loc[transactions_df['Transaction_ID'] == txn_data['transaction_id'], 'Is_Fraud'] = 2
            transactions_df.to_csv("pages/fraud_detect/transaction_history.csv", index=False)
            st.success("pages/fraud_detect/Transaction rejected successfully!")
            
        return True
    except Exception as e:
        st.error(f"Error processing transaction: {str(e)}")
        return False

# Check for email action
query_params = st.experimental_get_query_params()
if 'action' in query_params and 'data' in query_params:
    action = query_params['action'][0]
    encoded_data = query_params['data'][0]
    handle_transaction_action(action, encoded_data)
    st.experimental_rerun()

def handle_email_approval(transaction_data):
    """Handle transaction approval from email link"""
    try:
        # Parse transaction data
        txn = json.loads(transaction_data)
        
        # Update transaction status
        transactions_df.loc[transactions_df['Transaction_ID'] == txn['Transaction_ID'], 'Is_Fraud'] = 0
        
        # Update user balances
        users_df.loc[users_df['User_ID'] == txn['User_ID'], 'Initial_Balance'] -= txn['Amount']
        users_df.loc[users_df['User_ID'] == txn['Recipient_ID'], 'Initial_Balance'] += txn['Amount']
        
        # Save changes
        transactions_df.to_csv("pages/fraud_detect/transaction_history.csv", index=False)
        users_df.to_csv("pages/fraud_detect/user_profiles.csv", index=False)
        
        return True
    except Exception as e:
        print(f"Error handling approval: {str(e)}")
        return False

def preprocess_transaction(transaction):
    """Preprocess transaction data for model prediction."""
    # Convert categorical features to numerical
    location_map = {'USA': 0, 'UK': 1, 'Canada': 2, 'India': 3, 'Nigeria': 4, 'Russia': 5}
    device_map = {'Desktop': 0, 'Mobile': 1}
    behavior_map = {'Normal': 0, 'Abnormal': 1}
    
    # Create feature vector
    features = np.array([
        transaction['Amount'],
        location_map.get(transaction['Location'], 0),
        device_map.get(transaction['Device'], 0),
        behavior_map.get(transaction['Behavior'], 0)
    ])
    
    return features

def is_suspicious_transaction(transaction, user_data):
    """Determine if a transaction is suspicious based on multiple factors"""
    # Get user's base location
    base_location = user_data['Base_Location']
    
    # High-risk locations
    high_risk_locations = ['Nigeria', 'Russia']
    
    # Suspicious patterns
    suspicious_patterns = [
        # Location-based checks
        transaction['Location'] in high_risk_locations and transaction['Amount'] > 3000,
        transaction['Location'] != base_location and transaction['Amount'] > 5000,
        
        # Device-based checks
        transaction['Device'] == 'Mobile' and transaction['Amount'] > 10000,
        
        # Amount-based checks
        transaction['Amount'] > 20000,  # Very large transactions
        transaction['Amount'] < 1,  # Very small transactions
    ]
    
    # Return True if any suspicious pattern is detected
    return any(suspicious_patterns)

def get_recipient_risk_status(recipient_id):
    """Determine if a recipient is high-risk based on their location and transaction history"""
    # Specific high-risk users
    high_risk_users = ['U1002', 'U1004']
    
    if recipient_id in high_risk_users:
        return "High-Risk", "Recipient has high-risk transaction history"
    
    recipient_data = users_df[users_df["User_ID"] == recipient_id].iloc[0]
    recipient_txns = transactions_df[transactions_df["User_ID"] == recipient_id]
    
    # High-risk locations
    high_risk_locations = ['Nigeria', 'Russia']
    
    # Check if recipient is in high-risk location
    if recipient_data['Base_Location'] in high_risk_locations:
        return "High-Risk", "Recipient is located in a high-risk country"
    
    # Check recipient's transaction history
    fraud_count = recipient_txns["Is_Fraud"].sum()
    if fraud_count > 0:
        return "High-Risk", f"Recipient has {fraud_count} suspicious transactions"
    
    return "Safe", None

# User Dashboard
def user_dashboard(user_id):
    user_data = users_df[users_df["User_ID"] == user_id].iloc[0]
    user_txns = transactions_df[transactions_df["User_ID"] == user_id].sort_values("Timestamp", ascending=False)
    
    st.subheader(f"Account Overview: {user_data['Name']}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Account Balance", f"₹{user_data['Initial_Balance']:,.2f}")
        st.write(f"**User ID:** {user_id}")
        st.write(f"**Base Location:** {user_data['Base_Location']}")
    
    with col2:
        # Count suspicious transactions (Is_Fraud = 1)
        suspicious_count = len(user_txns[user_txns["Is_Fraud"] == 1])
        st.metric("Suspicious Transactions", suspicious_count)
    
    # Transfer Form
    with st.form("transfer_form"):
        st.subheader("Initiate Transfer")
        
        # Get recipient options
        recipient_options = users_df[users_df["User_ID"] != user_id]["User_ID"].values
        
        # Create selectbox with just User IDs
        selected_recipient = st.selectbox(
            "Recipient",
            options=recipient_options,
            index=0  # Default to first recipient
        )
        
        amount = st.number_input("Amount (₹)", min_value=1, max_value=100000)
        location = st.selectbox("Transaction Location", ["USA", "UK", "Canada", "India", "Nigeria", "Russia"])
        device = st.radio("Device Used", ["Mobile", "Desktop"])
        
        submitted = st.form_submit_button("Submit Transfer")
        
        if submitted:
            # Check recipient risk status
            risk_status, risk_reason = get_recipient_risk_status(selected_recipient)
            
            # Generate transaction
            new_txn = {
                "Transaction_ID": f"TXN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "User_ID": user_id,
                "Recipient_ID": selected_recipient,
                "Amount": amount,
                "Location": location,
                "Device": device,
                "Behavior": "Abnormal" if (location in ["Nigeria", "Russia"] and amount > 3000) else "Normal",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Is_Fraud": 1 if risk_status == "High-Risk" else 0  # Set to 1 for high-risk transactions
            }
            
            # Store transaction details in session state
            st.session_state.pending_transaction = new_txn
            st.session_state.risk_status = risk_status
            st.session_state.risk_reason = risk_reason
            st.session_state.amount = amount
            st.session_state.selected_recipient = selected_recipient
    
    # Handle transaction after form submission
    if 'pending_transaction' in st.session_state:
        txn = st.session_state.pending_transaction
        risk_status = st.session_state.risk_status
        risk_reason = st.session_state.risk_reason
        amount = st.session_state.amount
        selected_recipient = st.session_state.selected_recipient
        
        # Show transaction details
        st.info("🔄 Processing your transaction...")
        st.write("Transaction Details:")
        st.write(f"Amount: ₹{amount:,.2f}")
        st.write(f"Recipient: {selected_recipient}")
        st.write(f"Location: {txn['Location']}")
        st.write(f"Device: {txn['Device']}")
        
        # Show risk status
        if risk_status == "Safe":
            st.success("✅ Safe Recipient")
        else:
            st.error(f"🚨 High-Risk Recipient: {risk_reason}")
            st.warning("⚠️ This will be counted as a suspicious transaction!")
        
        # Show approve/reject buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve Transfer"):
                with st.spinner("Processing transfer..."):
                    try:
                        # Verify sufficient balance
                        sender_balance = users_df.loc[users_df["User_ID"] == user_id, "Initial_Balance"].iloc[0]
                        if sender_balance < amount:
                            st.error("❌ Insufficient balance for this transfer!")
                            return
                            
                        # Update balances
                        users_df.loc[users_df["User_ID"] == user_id, "Initial_Balance"] -= float(amount)
                        users_df.loc[users_df["User_ID"] == selected_recipient, "Initial_Balance"] += float(amount)
                        
                        # Create transaction record with recipient and risk status
                        new_transaction = {
                            "Transaction_ID": txn["Transaction_ID"],
                            "User_ID": user_id,
                            "Recipient_ID": selected_recipient,
                            "Amount": float(amount),
                            "Location": txn["Location"],
                            "Device": txn["Device"],
                            "Behavior": txn["Behavior"],
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Is_Fraud": 1 if risk_status == "High-Risk" else 0
                        }
                        
                        # Save transaction
                        transactions_df.loc[len(transactions_df)] = new_transaction
                        transactions_df.to_csv("pages/fraud_detect/transaction_history.csv", index=False)
                        users_df.to_csv("pages/fraud_detect/user_profiles.csv", index=False)
                        
                        # Show transfer success
                        st.success("✅ Transfer completed successfully!")
                        
                        # Send Gmail notification
                        with st.spinner("📧 Sending confirmation email..."):
                            try:
                                if email_system.send_alert(
                                    recipient_email=user_data['Email'],
                                    transaction=new_transaction
                                ):
                                    st.success("📧 Gmail confirmation sent successfully!")
                            except Exception as e:
                                st.warning(f"Transfer completed but couldn't send Gmail: {str(e)}")
                        
                        # Show updated transaction history immediately
                        st.subheader("Updated Transaction History")
                        # Get latest transactions including the new one
                        updated_txns = pd.read_csv("pages/fraud_detect/transaction_history.csv")
                        user_txns = updated_txns[
                            (updated_txns["User_ID"] == user_id) | 
                            (updated_txns["Recipient_ID"] == user_id)
                        ].copy()
                        
                        if not user_txns.empty:
                            # Add transaction type column
                            user_txns['Type'] = 'Sent'
                            user_txns.loc[user_txns['Recipient_ID'] == user_id, 'Type'] = 'Received'
                            
                            # Format amount with sign and add risk status
                            user_txns['Display_Amount'] = user_txns.apply(
                                lambda x: f"-₹{x['Amount']:,.2f}" if x['Type'] == 'Sent' else f"+₹{x['Amount']:,.2f}",
                                axis=1
                            )
                            
                            # Add Risk Status column
                            user_txns['Risk_Status'] = user_txns['Is_Fraud'].map({0: '✅ Safe', 1: '🚨 High-Risk'})
                            
                            # Display columns in order
                            display_columns = ['Transaction_ID', 'Type', 'Display_Amount', 'Location', 'Device', 'Risk_Status', 'Timestamp']
                            st.dataframe(
                                user_txns[display_columns].sort_values("Timestamp", ascending=False),
                                use_container_width=True
                            )
                        
                        # Clear the transaction
                        del st.session_state.pending_transaction
                        
                    except Exception as e:
                        st.error(f"Error processing transfer: {str(e)}")
        
        with col2:
            if st.button("Reject Transfer"):
                st.info("❌ Transfer rejected")
                del st.session_state.pending_transaction
                st.experimental_rerun()
                
    # Always show transaction history at the bottom
    st.subheader("Your Transaction History")
    user_txns = transactions_df[
        (transactions_df["User_ID"] == user_id) | 
        (transactions_df["Recipient_ID"] == user_id)
    ].copy()
    
    if not user_txns.empty:
        # Add transaction type column
        user_txns['Type'] = 'Sent'
        user_txns.loc[user_txns['Recipient_ID'] == user_id, 'Type'] = 'Received'
        
        # Format amount with sign and add risk status
        user_txns['Display_Amount'] = user_txns.apply(
            lambda x: f"-₹{x['Amount']:,.2f}" if x['Type'] == 'Sent' else f"+₹{x['Amount']:,.2f}",
            axis=1
        )
        
        # Add Risk Status column
        user_txns['Risk_Status'] = user_txns['Is_Fraud'].map({0: '✅ Safe', 1: '🚨 High-Risk'})
        
        # Display columns in order
        display_columns = ['Transaction_ID', 'Type', 'Display_Amount', 'Location', 'Device', 'Risk_Status', 'Timestamp']
        st.dataframe(
            user_txns[display_columns].sort_values("Timestamp", ascending=False),
            use_container_width=True
        )

# Main App
def main():
    st.title("🏦 Secure Banking Portal")
    
    # User selection with scrollable dropdown
    user_options = users_df["User_ID"].values
    user_id = st.selectbox(
        "Select Your Account",
        options=user_options,
        format_func=lambda x: f"{users_df[users_df['User_ID'] == x]['Name'].iloc[0]} ({x})",
        index=0  # Default to first user
    )
    
    if user_id:
        user_dashboard(user_id)

if __name__ == "__main__":
    main()