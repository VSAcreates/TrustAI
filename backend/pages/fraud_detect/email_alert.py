import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import base64
import json

class EmailAlertSystem:
    def __init__(self):
        load_dotenv()
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.sender_email = os.getenv('SENDER_EMAIL')
        self.sender_password = os.getenv('SENDER_PASSWORD')
        
        if not all([self.smtp_server, self.smtp_port, self.sender_email, self.sender_password]):
            raise ValueError("Missing email configuration. Please check .env file.")
        
    def send_alert(self, recipient_email, transaction):
        """Send an email alert about a suspicious transaction.
        
        Args:
            recipient_email (str): Email address of the recipient
            transaction (dict): Dictionary containing transaction details
        """
        if not recipient_email:
            raise ValueError("Recipient email is required")
            
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = "Transaction Confirmation"
            
            # Encode transaction data for approval link
            txn_data = {
                'transaction_id': transaction['Transaction_ID'],
                'user_id': transaction['User_ID'],
                'recipient_id': transaction['Recipient_ID'],
                'amount': transaction['Amount']
            }
            encoded_data = base64.b64encode(json.dumps(txn_data).encode()).decode()
            
            # Create HTML message
            html = f"""
            <html>
                <body style="font-family: Arial, sans-serif; padding: 20px;">
                    <h2 style="color: #4CAF50;">✅ Transaction Confirmation</h2>
                    
                    <div style="background-color: #f8f8f8; padding: 15px; border-radius: 5px; margin: 10px 0;">
                        <h3>Transaction Details:</h3>
                        <p><strong>Amount:</strong> ${transaction.get('Amount', 0):,.2f}</p>
                        <p><strong>Recipient ID:</strong> {transaction.get('Recipient_ID', 'Unknown')}</p>
                        <p><strong>Location:</strong> {transaction.get('Location', 'Unknown')}</p>
                        <p><strong>Device:</strong> {transaction.get('Device', 'Unknown')}</p>
                        <p><strong>Time:</strong> {transaction.get('Timestamp', 'Unknown')}</p>
                    </div>
                    
                    <p>Your transaction has been processed successfully.</p>
                    
                    <p style="color: #666; font-size: 12px; margin-top: 20px;">
                        If you did not initiate this transaction, please contact our support team immediately.
                    </p>
                </body>
            </html>
            """
            
            # Create plain text version
            text = f"""
            ✅ Transaction Confirmation
            
            Transaction Details:
            - Amount: ${transaction.get('Amount', 0):,.2f}
            - Recipient ID: {transaction.get('Recipient_ID', 'Unknown')}
            - Location: {transaction.get('Location', 'Unknown')}
            - Device: {transaction.get('Device', 'Unknown')}
            - Time: {transaction.get('Timestamp', 'Unknown')}
            
            Your transaction has been processed successfully.
            
            If you did not initiate this transaction, please contact our support team immediately.
            """
            
            # Attach both versions
            msg.attach(MIMEText(text, 'plain'))
            msg.attach(MIMEText(html, 'html'))
            
            # Create SMTP session
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                print(f"Attempting to login with email: {self.sender_email}")  # Debug log
                server.login(self.sender_email, self.sender_password)
                print("Login successful")  # Debug log
                server.send_message(msg)
                print("Message sent successfully")  # Debug log
                return True
                
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            raise  # Re-raise the exception to handle it in the app 