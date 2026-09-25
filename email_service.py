
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("SMTP_EMAIL")
        self.sender_password = os.getenv("SMTP_PASSWORD")
        self.is_configured = all([self.sender_email, self.sender_password])
    
    async def send_password_reset_email(self, recipient_email: str, reset_code: str) -> bool:
        """Send password reset code to user's email"""
        
        if not self.is_configured:
            print(f"❌ Email not configured. Code for {recipient_email}: {reset_code}")
            return False
        
        try:
            # Validate email format
            if '@' not in recipient_email or '.' not in recipient_email:
                print(f"❌ Invalid email address: {recipient_email}")
                return False

            # Create message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = recipient_email
            message["Subject"] = "Agronomist AI - Password Reset Code"
            
            # HTML email body
            body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background: #10B981; color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0; }}
                    .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                    .code {{ font-size: 32px; font-weight: bold; color: #10B981; text-align: center; letter-spacing: 5px; margin: 20px 0; }}
                    .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🔐 Agronomist AI</h1>
                        <p>Password Reset Request</p>
                    </div>
                    <div class="content">
                        <p>Hello,</p>
                        <p>You requested a password reset for your Agronomist AI account.</p>
                        
                        <div class="code">{reset_code}</div>
                        
                        <p>Enter this code in the application to reset your password.</p>
                        <p><strong>This code will expire in 15 minutes.</strong></p>
                        
                        <p>If you didn't request this reset, please ignore this email.</p>
                        
                        <p>Best regards,<br>
                        <strong>Agronomist AI Team</strong></p>
                    </div>
                    <div class="footer">
                        <p>This is an automated message. Please do not reply to this email.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            message.attach(MIMEText(body, "html"))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
            
            print(f"✅ Password reset email sent to {recipient_email}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            print(f"❌ SMTP Authentication failed: {e}")
            print("💡 Please check your SMTP_EMAIL and SMTP_PASSWORD in .env file")
            return False
        except smtplib.SMTPException as e:
            print(f"❌ SMTP error occurred: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error sending email: {e}")
            return False

email_service = EmailService()