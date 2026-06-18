import base64
import email
from email import message_from_bytes
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func
import re
from app.config import SessionLocal
from app.models import Email, User  # Make sure User is imported
from app.categorization import categorize_email
# Set up logging
logger = logging.getLogger(__name__)


class EmailProcessor:
    def __init__(self, credentials):
        """
        Initializes the EmailProcessor with user credentials.
        """
        self.credentials = credentials
        self.service = build('gmail', 'v1', credentials=self.credentials)

        logger.info("Loading SentenceTransformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully.")

    def fetch_and_save_emails(self, user: User, limit=50) -> List[Dict[str, Any]]:
        """
        Fetches emails from Gmail for a specific user, saves them to the DB
        with the user's ID, and returns a list of email dictionaries.
        """
        db_session = SessionLocal()
        fetched_email_objects = []
        try:
            results = self.service.users().messages().list(
                userId='me', maxResults=limit, q='-in:drafts -in:chats'
            ).execute()
            messages = results.get('messages', [])

            if not messages:
                logger.info("No new emails found.")
                return []

            for msg_info in messages:
                existing_email = db_session.query(Email).filter(Email.id == msg_info['id']).first()
                if existing_email:
                    continue

                msg = self.service.users().messages().get(userId='me', id=msg_info['id'], format='raw').execute()
                raw_email = base64.urlsafe_b64decode(msg['raw'].encode('ASCII'))
                email_message = message_from_bytes(raw_email)

                subject = self.decode_header(email_message['subject'])
                sender = self.decode_header(email_message['from'])
                body = self.get_email_body(email_message)
                recipient_header = email_message.get('to')
                recipient = self.decode_header(recipient_header) if recipient_header else 'me'

                try:
                    timestamp_str = email_message['date']
                    timestamp_dt = datetime.strptime(timestamp_str, '%a, %d %b %Y %H:%M:%S %z').astimezone(timezone.utc)
                except Exception:
                    timestamp_dt = datetime.now(timezone.utc)

                categories = categorize_email({"subject": subject, "body": body}) or []
                categories = [c.strip().title() for c in categories if c.strip()]
                if not categories:
                    categories = ["Uncategorized"]
                category_str = ", ".join(sorted(set(categories)))

                new_email = Email(
                    id=msg['id'],
                    sender=sender, recipient=recipient, subject=subject, body=body,
                    timestamp=timestamp_dt, category=category_str,
                    status='unread' if 'UNREAD' in msg.get('labelIds', []) else 'read',
                    # --- THIS IS THE CRITICAL FIX ---
                    user_id=user.id
                )
                db_session.add(new_email)
                fetched_email_objects.append(new_email)

            db_session.commit()
            logger.info(f"Successfully saved {len(fetched_email_objects)} new emails for user {user.email}.")

            # Convert objects to dictionaries before the session closes
            email_dicts = [
                {
                    "id": e.id, "subject": e.subject, "sender": e.sender,
                    "body": e.body, "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S") if e.timestamp else None,
                    "category": e.category, "priority": getattr(e, 'priority', 'Normal'),
                    "summary": getattr(e, 'summary', None), "ai_response": getattr(e, 'ai_response', None)
                }
                for e in fetched_email_objects
            ]
            return email_dicts

        except SQLAlchemyError as e:
            db_session.rollback()
            logger.error(f"Database error for user {user.email}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching emails for user {user.email}: {e}")
            return []
        finally:
            db_session.close()

    def get_email_stats(self, user: User) -> Dict[str, Any]:
        """
        Analyzes emails for a specific user and returns key statistics.
        """
        stats = {}
        db_session = SessionLocal()
        try:
            # Gmail stats are user-specific by default
            unread_results = self.service.users().labels().get(userId='me', id='UNREAD').execute()
            stats['unread'] = unread_results.get('messagesUnread', 0)
            threads_results = self.service.users().threads().list(userId='me').execute()
            stats['threads'] = len(threads_results.get('threads', []))

            # Database queries must be filtered by user_id
            action_items_count = db_session.query(Email).filter(Email.user_id == user.id, Email.category.ilike('%Action Item%')).count()
            stats['action_items'] = action_items_count

            category_rows = db_session.query(Email.category).filter(Email.user_id == user.id).all()
            flat_counts = {}
            for (cat_str,) in category_rows:
                if not cat_str: cat_str = "Uncategorized"
                for cat in [c.strip() for c in cat_str.split(",")]:
                    if cat: flat_counts[cat] = flat_counts.get(cat, 0) + 1
            stats['categories'] = flat_counts or {"Uncategorized": 0}

            stats['pending'] = db_session.query(Email).filter(Email.user_id == user.id, Email.status == "todo").count()
        except Exception as e:
            logger.error(f"Error getting stats for user {user.email}: {e}")
            return {} # Return empty dict on error
        finally:
            db_session.close()

        stats['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return stats

    # --- Helper methods (no changes needed) ---
    def decode_header(self, header):
        if header is None: return ""
        decoded_parts = []
        for part, encoding in email.header.decode_header(header):
            if isinstance(part, bytes):
                decoded_parts.append(part.decode(encoding or 'utf-8', errors='ignore'))
            else:
                decoded_parts.append(part)
        return "".join(decoded_parts)

    def get_email_body(self, msg):
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain' and 'attachment' not in part.get('Content-Disposition', ''):
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
        else:
            if msg.get_content_type() == 'text/plain':
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        return body

    # --- Data fetching methods (updated to filter by user) ---
    def get_action_items(self, user: User, status_filter: str = "All", limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        session = SessionLocal()
        try:
            query = (
                session.query(Email)
                .filter(Email.user_id == user.id, Email.category.ilike("%Action Item%"))
                .order_by(Email.timestamp.desc())
            )
            if status_filter != "All":
                query = query.filter(Email.status.ilike(status_filter))
            emails = query.offset(offset).limit(limit).all()
            return [
                {
                    "id": e.id, "subject": e.subject, "sender": e.sender,
                    "recipient": e.recipient, "date": e.timestamp.strftime("%Y-%m-%d"),
                    "status": e.status, "body": e.body, "summary": getattr(e, "summary", None),
                    "ai_response": getattr(e, "ai_response", None),
                } for e in emails
            ]
        except Exception as e:
            logger.error(f"Error fetching action items for user {user.email}: {e}")
            return []
        finally:
            session.close()

    def get_todo_emails(self, user: User, status_filter: str = "All", limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        session = SessionLocal()
        try:
            query = (
                session.query(Email)
                .filter(Email.user_id == user.id, Email.status == "todo")
                .order_by(Email.timestamp.desc())
            )
            if status_filter != "All":
                query = query.filter(Email.status.ilike(status_filter))
            emails = query.offset(offset).limit(limit).all()

            todo_items = []
            for email_obj in emails:
                deadline = None
                deadline_match = re.search(r"(?:due|deadline)[:\s]*([A-Za-z0-9 ,/-]+)", f"{email_obj.subject} {email_obj.body}", re.IGNORECASE)
                if deadline_match:
                    deadline = deadline_match.group(1).strip()

                todo_items.append({
                    "id": email_obj.id, "subject": email_obj.subject, "sender": email_obj.sender,
                    "recipient": email_obj.recipient, "date": email_obj.timestamp.strftime("%Y-%m-%d"),
                    "status": email_obj.status, "deadline": deadline, "body": email_obj.body,
                    "summary": getattr(email_obj, "summary", None), "ai_response": getattr(email_obj, "ai_response", None)
                })
            return todo_items
        except Exception as e:
            logger.error(f"Error fetching TODO emails for user {user.email}: {e}")
            return []
        finally:
            session.close()

