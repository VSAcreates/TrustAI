import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st

class EmailAlertSystem:
    def __init__(self):
        self.sender = "fraud.alerts@yourbank.com"
        self.password = "your_app_password"  # Use environment variables in production
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.admin_email = "security@yourbank.com"

    def send_alert(self, transaction_data, recipient_email=None):
        """Send fraud alert to admin and optionally to user"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = "🚨 Suspicious Transaction Alert"
            msg['From'] = self.sender
            msg['To'] = self.admin_email
            
            if recipient_email:
                msg['Cc'] = recipient_email

            # HTML email content
            html = f"""
            <html>
              <body>
                <h2 style="color: #d9534f;">Fraud Alert</h2>
                <p><strong>Transaction ID:</strong> {transaction_data['Transaction_ID']}</p>
                <p><strong>User ID:</strong> {transaction_data['User_ID']}</p>
                <p><strong>Amount:</strong> ${transaction_data['Amount']:,.2f}</p>
                <p><strong>Location:</strong> {transaction_data['Location']}</p>
                <p><strong>Device:</strong> {transaction_data['Device']}</p>
                <p><strong>Behavior:</strong> {transaction_data['Behavior']}</p>
                <br>
                <p>Please review immediately at <a href="http://yourbank.com/security">Security Dashboard</a></p>
              </body>
            </html>
            """

            msg.attach(MIMEText(html, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            st.error(f"Email failed: {str(e)}")
            return False