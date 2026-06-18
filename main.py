import streamlit as st
import html
from datetime import timezone
import os
import json
from bs4 import BeautifulSoup
from app.ai_services import analyze_email_content, initialize_models
import pandas as pd
from datetime import datetime
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import base64
import re
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
import altair as alt
import psycopg2
import uuid
from app.rag import EmbedData, MilvusVDB, Retriever, RAG, VECTOR_DIMENSION, MILVUS_COLLECTION_NAME, MILVUS_DB_FILE
from app.categorization import categorize_email


class CustomHTTPAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)


# Initialize AI models
initialize_models()

# Page configuration
st.set_page_config(page_title="Mail Mentor", layout="wide")
st.set_option('client.showErrorDetails', True)


@st.cache_resource
def get_db_resources():
    from app.models import Email
    from app.config import SessionLocal
    from app.email_processor import EmailProcessor
    from app.auth import User, authenticate_user
    return Email, SessionLocal, EmailProcessor, User, authenticate_user


Email, SessionLocal, EmailProcessor, User, authenticate_user = get_db_resources()

if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'email_processor' not in st.session_state:
    session = requests.Session()
    session.mount('https://', CustomHTTPAdapter())
    st.session_state['email_processor'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'
if 'email_cache' not in st.session_state:
    st.session_state['email_cache'] = {}
if 'last_cache_update' not in st.session_state:
    st.session_state['last_cache_update'] = None
if 'emails_per_page' not in st.session_state:
    st.session_state['emails_per_page'] = 20
if 'rag_pipeline' not in st.session_state:
    st.session_state['rag_pipeline'] = None


# --- CORRECTED FUNCTION ---
def fetch_emails_as_dataframe(user_id: int):
    """
    Fetches emails for a SPECIFIC user from the PostgreSQL database.
    """
    DATABASE_URL = "postgresql://postgres:secret@localhost:5432/mailmentor"
    conn = None
    try:

        conn = psycopg2.connect(DATABASE_URL)
        # Add a WHERE clause to filter emails by the logged-in user's ID.
        query = "SELECT id, subject, sender, body, timestamp, summary, ai_response FROM emails WHERE user_id = %s ORDER BY timestamp DESC"

        # Pass the user_id safely as a parameter to the query.
        df = pd.read_sql_query(query, conn, params=(user_id,))

        return df
    except (Exception, psycopg2.DatabaseError) as error:
        st.error(f"Error fetching data from PostgreSQL: {error}")
        return pd.DataFrame()
    finally:
        if conn is not None:
            conn.close()


# --- CORRECTED FUNCTION ---
def build_rag_index():
    """
    Builds a user-specific Smart Search index using only their emails.
    """
    # 1. Verify that a user is logged in.
    if 'user' not in st.session_state or not st.session_state.user:
        st.warning("No user logged in. Cannot build search index.")
        return

    # 2. Get user ID for data isolation.
    user_id = st.session_state.user.id
    collection_name = f"{MILVUS_COLLECTION_NAME}_user_{user_id}"
    db_file_name = f"{MILVUS_DB_FILE.split('.')[0]}_user_{user_id}.db"

    with st.spinner(f"Building Smart Search index for {st.session_state.user.email}..."):
        # 3. Fetch emails for THIS USER ONLY by passing the user_id.
        df = fetch_emails_as_dataframe(user_id=user_id)
        if df.empty:
            st.warning("No emails found for your account to build the search index.")
            return

        emails_for_rag = [
            {
                "id": str(row.get('id', uuid.uuid4())),
                "sender": row.get('sender', 'Unknown Sender'),
                "subject": row.get('subject', 'No Subject'),
                "body": row.get('body', 'No Body')
            }
            for _, row in df.iterrows()
        ]
        st.info(f"Preparing to index {len(emails_for_rag)} emails.")

        # 4. Initialize RAG components with user-specific names.
        embedder = EmbedData()
        vector_db = MilvusVDB(
            collection_name=collection_name,
            vector_dim=VECTOR_DIMENSION,
            db_file=db_file_name
        )

        st.info(f"Setting up secure collection: {collection_name}")
        vector_db.create_collection()

        st.info("Generating vector embeddings...")
        email_contexts = [f"From: {e['sender']}\nSubject: {e['subject']}\n\n{e['body']}" for e in emails_for_rag]
        email_embeddings = embedder.embed(email_contexts)

        st.info("Ingesting data into the vector database...")
        vector_db.ingest_data(emails_for_rag, email_embeddings)

        retriever = Retriever(vector_db, embedder)
        rag_pipeline = RAG(retriever)
        st.session_state['rag_pipeline'] = rag_pipeline

        st.success("✅ Your private Smart Search index is ready!")


def apply_theme():
    if 'theme' in st.session_state:
        theme = st.session_state.theme
        if theme == "Dark":
            st.markdown("""
                <style>
                    :root {
                        --primary-color: #1f1f1f;
                        --background-color: #121212;
                        --secondary-background-color: #1f1f1f;
                        --text-color: #ffffff;
                    }
                    .stApp {
                        background-color: var(--background-color);
                        color: var(--text-color);
                    }
                </style>
            """, unsafe_allow_html=True)
        elif theme == "Light":
            st.markdown("""
                <style>
                    :root {
                        --primary-color: #ffffff;
                        --background-color: #ffffff;
                        --secondary-background-color: #f0f2f6;
                        --text-color: #000000;
                    }
                    .stApp {
                        background-color: var(--background-color);
                        color: var(--text-color);
                    }
                </style>
            """, unsafe_allow_html=True)


def apply_custom_css():
    st.markdown("""
        <style>
        .ai-analysis-container {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid rgba(49, 51, 63, 0.2);
        }
        .ai-summary {
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(49, 51, 63, 0.2);
        }
        .ai-response {
            margin-top: 10px;
        }
        .rag-answer {
            background-color: #1E222A;
            border-left: 5px solid #3498DB;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
        }
        .rag-source {
            background-color: #1A1C24;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        </style>

        <style>
            /* Global Styles */
            :root {
                --primary-color: #2E86C1;
                --accent-color: #3498DB;
                --background-color: #0E1117;
                --secondary-bg: #1A1C24;
                --text-color: #F8F9FA;
                --border-color: rgba(255, 255, 255, 0.1);
            }

            /* Main Container Styling */
            .block-container {
                padding: 3rem 5rem;
            }

            /* Sidebar Styling */
            section[data-testid="stSidebar"] {
                background-color: var(--secondary-bg) !important;
                border-right: 1px solid var(--border-color);
            }
            section[data-testid="stSidebar"] .stButton button {
                width: 100%;
                margin-bottom: 0.5rem;
                background-color: transparent;
                border: 1px solid var(--border-color);
                color: var(--text-color);
                transition: all 0.3s ease;
                border-radius: 0.5rem;
                padding: 0.5rem 1rem;
            }
            section[data-testid="stSidebar"] .stButton button:hover {
                background-color: var(--primary-color);
                border-color: var(--primary-color);
            }

            /* Card Styling */
            div[data-testid="stExpander"] {
                background-color: var(--secondary-bg);
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                overflow: hidden;
            }

            /* Metric Cards */
            div[data-testid="stMetric"] {
                background-color: var(--secondary-bg);
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid var(--border-color);
            }

            /* Tabs Styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2rem;
                background-color: transparent;
            }
            .stTabs [data-baseweb="tab"] {
                height: 3rem;
                color: var(--text-color);
                border-radius: 0.5rem;
            }
            .stTabs [aria-selected="true"] {
                background-color: var(--primary-color);
                border-radius: 0.5rem;
            }

            /* Button Styling */
            .stButton>button {
                border-radius: 0.5rem;
                transition: all 0.3s ease;
            }
            .stButton>button[data-baseweb="button"] {
                background-color: var(--primary-color);
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }

            /* Input Fields */
            .stTextInput>div>div>input,
            .stTextArea>div>div>textarea {
                background-color: var(--secondary-bg);
                border-color: var(--border-color);
                color: var(--text-color);
                border-radius: 0.5rem;
            }

            /* Selectbox Styling */
            .stSelectbox>div>div>div {
                background-color: var(--secondary-bg);
                border-color: var(--border-color);
                color: var(--text-color);
                border-radius: 0.5rem;
            }

            /* Progress Bar */
            .stProgress>div>div>div>div {
                background-color: var(--primary-color);
            }

            /* Alert/Info Boxes */
            .stAlert {
                background-color: var(--secondary-bg);
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
            }

            /* Table Styling */
            .stTable {
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
                overflow: hidden;
            }
            .stTable thead tr th {
                background-color: var(--secondary-bg);
                color: var(--text-color);
            }
            .stTable tbody tr:nth-of-type(odd) {
                background-color: rgba(255, 255, 255, 0.05);
            }

            /* Remove unwanted elements */
            .css-6qob1r.e1fqkh3o3 {display: none !important;}
            .viewerBadge_container__1QSob {display: none !important;}
        </style>
    """, unsafe_allow_html=True)


def init_gmail_connection(user, db_session):
    """
    Initializes the Gmail connection and creates the EmailProcessor.
    The vector store population step has been removed.
    """
    try:
        # --- Step 1: Authenticate and get credentials ---
        credentials = None
        if user.gmail_credentials:
            creds_dict = json.loads(user.gmail_credentials)
            credentials = Credentials.from_authorized_user_info(creds_dict)

            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
                user.gmail_credentials = credentials.to_json()
                db_session.add(user)
                db_session.commit()
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'config/credentials.json',
                ['https://www.googleapis.com/auth/gmail.readonly']
            )
            credentials = flow.run_local_server(port=0)
            user.gmail_credentials = credentials.to_json()
            db_session.add(user)
            db_session.commit()

        if not credentials:
            st.error("Failed to obtain Gmail credentials.")
            return False

        # --- Step 2: Create the EmailProcessor ---
        email_processor = EmailProcessor(credentials)
        st.session_state['email_processor'] = email_processor

        st.success("Successfully connected to your Gmail account!")

        return True

    except Exception as e:
        st.error(f"Failed to connect to Gmail and initialize: {str(e)}")
        return False


def show_login():
    """
    Handles the user login process, including authentication, Gmail initialization,
    and setting a flag for a one-time dashboard refresh.
    """
    if not init_db_connection():
        st.error("Cannot connect to the database. Please try again later.")
        return

    st.header("Login to MailMentor")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            session = SessionLocal()
            try:
                user = authenticate_user(email, password)
                if user:
                    st.session_state['user'] = user
                    st.session_state['authenticated'] = True

                    if init_gmail_connection(user, session):
                        build_rag_index()

                        st.session_state['page'] = 'Dashboard'
                        st.session_state['first_load_refresh_needed'] = True
                        st.rerun()
                        return
                    else:
                        st.error("Failed to connect to Gmail. Please check your credentials or permissions.")
                else:
                    st.error("Invalid email or password.")
            except Exception as e:
                st.error(f"An error occurred during login: {e}")
            finally:
                session.close()

    if st.button("Don't have an account? Register"):
        st.session_state['page'] = 'register'
        st.rerun()
        return


def show_register():
    if not init_db_connection():
        st.error("Cannot connect to database. Please try again later.")
        return

    st.header("Register for MailMentor")
    with st.form("register_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Register")

        if submit and name and email and password:
            session = SessionLocal()
            try:
                existing_user = session.query(User).filter(User.email == email).first()
                if existing_user:
                    st.error("Email already registered")
                    return

                user = User(name=name, email=email)
                user.set_password(password)
                session.add(user)
                session.commit()
                session.refresh(user)

                st.info("Clearing previous user's data from the database...")
                session.query(Email).delete()
                session.commit()
                st.info("Previous data cleared successfully.")

                if init_gmail_connection(user, session):
                    st.session_state['user'] = user
                    st.session_state['authenticated'] = True
                    build_rag_index()
                    st.session_state['page'] = 'Dashboard'
                    st.rerun()
                else:
                    st.error("Failed to connect to Gmail. Please try again.")
            except Exception as e:
                st.error(f"Registration failed: {str(e)}")
            finally:
                session.close()

    if st.button("Already have an account? Login"):
        st.session_state['page'] = 'login'
        st.rerun()


from sqlalchemy.orm import undefer


@st.cache_data(ttl=300)
def fetch_cached_stats(user_id: int):
    # Pass the user object from session state
    user = st.session_state.get('user')
    if not user or user.id != user_id:  # Security check
        return {}
    return st.session_state.email_processor.get_email_stats(user)


@st.cache_data(ttl=300)
def fetch_cached_emails(limit=20, offset=0):
    if not st.session_state.get('user'):
        return []
    user_id = st.session_state.user.id

    cache_key = f"emails_user_{user_id}_{limit}_{offset}"

    if cache_key in st.session_state['email_cache']:
        return st.session_state['email_cache'][cache_key]

    try:
        session = SessionLocal()
        db_objects = (
            session.query(Email)
            .filter(Email.user_id == user_id)
            .order_by(Email.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        email_dicts = [
            {
                "id": e.id, "subject": e.subject, "sender": e.sender,
                "body": e.body, "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S") if e.timestamp else None,
                "category": e.category, "priority": e.priority,
                "summary": e.summary, "ai_response": e.ai_response
            }
            for e in db_objects
        ]
        session.close()

        if email_dicts:
            st.session_state['email_cache'][cache_key] = email_dicts
            return email_dicts

        user = st.session_state.user
        emails_from_api = st.session_state.email_processor.fetch_and_save_emails(user=user, limit=limit)

        if emails_from_api:  # This is already a list of dicts
            st.session_state['email_cache'][cache_key] = emails_from_api
            return emails_from_api

        return []

    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        if 'session' in locals() and session.is_active:
            session.close()
        return []


@st.cache_data(ttl=300)
def filter_emails(emails):
    return [
        email for email in emails
        if not any(x in (email.get('sender', '') or '').lower() for x in ['no-reply', 'google', 'ads', 'notification'])
    ]


def create_time_series_chart(df):
    return alt.Chart(df).mark_line().encode(
        x=alt.X(df.columns[0], title='Date'),
        y=alt.Y(df.columns[1], title='Count')
    )


def create_sender_chart(df):
    return alt.Chart(df).mark_bar().encode(
        x=alt.X(df.columns[0], title='Sender'),
        y=alt.Y(df.columns[1], title='Count')
    )


def show_email_stats():
    if not st.session_state.email_processor:
        st.warning("Please connect your Gmail account first")
        return

    try:
        stats = fetch_cached_stats(user_id=st.session_state.user.id)
        if not stats or all(not data for data in stats.values()):
            st.warning("No email data available for visualization")
            return

        if stats.get('by_date'):
            chart_data = pd.DataFrame(stats['by_date'])
            if not chart_data.empty:
                st.altair_chart(create_time_series_chart(chart_data))

        if stats.get('by_sender'):
            sender_data = pd.DataFrame(stats['by_sender'])
            if not sender_data.empty:
                st.altair_chart(create_sender_chart(sender_data))

    except Exception as e:
        st.error(f"Error fetching email statistics: {str(e)}")
        print(f"Detailed error: {e}")


def show_recent_emails():
    st.header("Recent Emails")

    if not st.session_state.email_processor:
        st.warning("Please connect your Gmail account first")
        return

    emails = fetch_cached_emails(limit=10)

    for email in emails:
        # Categorize email
        categories = categorize_email(email)
        category_str = ", ".join(categories)

        with st.expander(f"📧 {email.get('subject', 'No Subject')}"):
            priority = email.get('priority', 'Normal')
            priority_color = {
                'High': '#f44336',
                'Normal': '#2196f3',
                'Low': '#4caf50'
            }.get(priority, '#2196f3')

            st.markdown(f"""
                <div style='border-left: 4px solid {priority_color}; padding-left: 1rem;'>
                    <p><strong>From:</strong> {email.get('sender', 'Unknown')}</p>
                    <p><strong>Date:</strong> {email.get('timestamp', 'Unknown')}</p>
                    <p><strong>Priority:</strong> {priority}</p>
                    <p><strong>Categories:</strong> {category_str}</p>
                </div>
            """, unsafe_allow_html=True)

            content = email.get('body', 'No content available')
            content = html.unescape(content)
            content = content.replace('<', '&lt;').replace('>', '&gt;')

            if len(content) > 500:
                st.markdown(content[:500])
                st.markdown("**Read more:**")
                st.markdown(content[500:])
            else:
                st.markdown(content)


def show_dashboard():
    # One-time refresh logic
    if st.session_state.get('first_load_refresh_needed', False):
        st.session_state['first_load_refresh_needed'] = False
        st.rerun()

    # Automated search indexing
    if 'index_built' not in st.session_state:
        st.session_state.index_built = False
    if not st.session_state.index_built:
        build_rag_index()
        st.session_state.index_built = True

    st.header("Dashboard")

    # Action Buttons
    col1, col2, col_spacer = st.columns([1, 2, 7])
    with col1:
        if st.button("🔄 Refresh"):
            with st.spinner("Fetching latest emails..."):
                user = st.session_state.user
                new_emails = st.session_state.email_processor.fetch_and_save_emails(user=user, limit=50)

            fetch_cached_stats.clear()
            fetch_cached_emails.clear()
            st.session_state.index_built = False

            if new_emails:
                st.success(f"Fetched {len(new_emails)} new emails ✅")
            else:
                st.info("No new emails found.")
            st.rerun()

    with col2:
        # --- CORRECTED CALL ---
        user_id = st.session_state.user.id
        df_for_download = fetch_emails_as_dataframe(user_id=user_id)
        if not df_for_download.empty:
            csv_data = df_for_download.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Emails as CSV",
                data=csv_data,
                file_name='emails_export.csv',
                mime='text/csv',
            )

    with st.spinner("Loading dashboard..."):
        stats = fetch_cached_stats(user_id=st.session_state.user.id)

        st.markdown("### 📊 Overview")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        metrics = {
            m_col1: ("📥", stats.get('unread', 0), "Unread Emails"),
            m_col2: ("✅", stats.get('action_items', 0), "Action Items"),
            m_col3: ("🔄", stats.get('threads', 0), "Active Threads"),
            m_col4: ("⏳", stats.get('pending', 0), "Pending Responses")
        }
        for col, (icon, value, label) in metrics.items():
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<h3 style='margin:0'>{icon} {value}</h3>"
                    f"<p style='margin:0;color:gray'>{label}</p></div>",
                    unsafe_allow_html=True
                )

        st.markdown("### 📈 Email Categories")
        categories = stats.get('categories', {})
        if categories:
            df = pd.DataFrame(list(categories.items()), columns=['Category', 'Count'])
            st.bar_chart(df.set_index('Category'))
        else:
            st.info("No category data available to display a chart.")

        tab1, tab2 = st.tabs(["📧 Recent Emails", "🤖 AI Summaries & Responses"])

        emails = fetch_cached_emails(limit=50)
        filtered_emails = filter_emails(emails)

        with tab1:
            st.info("📝 Note: Promotional and notification emails are automatically filtered out from this view.")
            for email in filtered_emails[:st.session_state.emails_per_page]:
                subject = html.escape(email.get('subject', 'No Subject'))
                sender = html.escape(email.get('sender', 'Unknown Sender'))
                timestamp = email.get('timestamp', 'Unknown Date')
                category = email.get('category', 'General')
                priority = email.get('priority', 'Normal')
                body = email.get('body', 'No content available')

                priority_color = {'High': '#f44336', 'Normal': '#2196f3', 'Low': '#4caf50'}.get(priority, '#2196f3')

                with st.expander(f"📧 {subject}"):
                    st.markdown(f"""
                        <div style='border-left: 4px solid {priority_color}; padding-left: 1rem;'>
                            <p><strong>From:</strong> {sender}</p>
                            <p><strong>Date:</strong> {timestamp}</p>
                            <p><strong>Category:</strong> {category}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown("---")
                    st.text(body)

        with tab2:
            st.info("Click 'Generate AI Analysis' on any email to process its content with AI.")
            for email in filtered_emails[:st.session_state.emails_per_page]:
                subject = html.escape(email.get('subject', 'No Subject'))

                with st.container():
                    st.markdown(f"#### {subject}")
                    if st.button("Analyze with AI", key=f"generate_ai_{email.get('id')}"):
                        with st.spinner(f"Generating AI analysis for '{subject}'..."):
                            summary, response = analyze_email_content(email.get('subject', ''), email.get('body', ''))
                            email['summary'] = summary
                            email['ai_response'] = response

                            try:
                                session = SessionLocal()
                                db_record = session.query(Email).filter(Email.id == email.get('id')).first()
                                if db_record:
                                    db_record.summary = summary
                                    db_record.ai_response = response
                                    session.commit()
                            except Exception as e:
                                st.error(f"Error saving AI analysis to DB: {e}")
                            finally:
                                session.close()
                        st.rerun()

                    if email.get('summary') or email.get('ai_response'):
                        summary_html = html.escape(email.get('summary', 'No summary available'))
                        response_html = html.escape(email.get('ai_response', 'No response available'))
                        st.markdown(f"""
                            <div class='ai-analysis-container'>
                                <div class='ai-summary'>
                                    <h5>🤖 AI Summary</h5>
                                    <p>{summary_html}</p>
                                </div>
                                <div class='ai-response'>
                                    <h5>💡 Suggested Response</h5>
                                    <p>{response_html}</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    st.divider()
        st.text(f"Last Updated: {stats.get('last_updated', 'N/A')}")


def authenticate_gmail():
    try:
        SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json',
            SCOPES
        )
        credentials = flow.run_local_server(port=0)
        st.session_state.email_processor = EmailProcessor(credentials)
        st.session_state.authenticated = True
        return True
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return False


def init_db_connection():
    try:
        session = SessionLocal()
        from sqlalchemy import text
        session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return False
    finally:
        session.close()


def show_inbox():
    st.header("Inbox")
    if not st.session_state.email_processor:
        st.warning("Please connect your Gmail account first")
        return

    emails = fetch_cached_emails(limit=50)
    if not emails:
        st.info("No emails found")
        return

    for email in emails:
        with st.expander(f"📧 {html.escape(email.get('subject', 'No Subject'))}"):
            st.markdown(f"**From:** {html.escape(email.get('sender', 'Unknown'))}")
            st.markdown(f"**Date:** {email.get('timestamp', 'Unknown')}")
            st.markdown(f"**Priority:** {email.get('priority', 'Normal')}")
            st.markdown("---")
            st.markdown(html.escape(email.get('body', 'No content')))


@st.cache_data(ttl=300)
def fetch_cached_action_items(status_filter="All"):
    user = st.session_state.user
    return st.session_state.email_processor.get_action_items(user=user, status_filter=status_filter)


def show_action_items():
    st.header("Action Items")
    if not st.session_state.email_processor:
        st.warning("Please connect your Gmail account first")
        return
    status_filter = st.selectbox("Filter by Status", ["All", "Pending", "Completed"])
    action_items = fetch_cached_action_items(status_filter)
    if not action_items:
        st.info("No action items found")
        return
    for item in action_items:
        with st.expander(f"📋 {item['subject']}"):
            st.write(f"**From:** {item['sender']}")
            st.write(f"**Date:** {item.get('date', 'No date')}")
            st.write(f"**Priority:** {item.get('priority', 'Normal')}")
            st.write(f"**Status:** {item.get('status', 'Pending')}")
            st.write(f"**Content:** {item.get('body', 'No content available')}")
            st.write(f"**AI Summary:** {item.get('summary', 'No summary available')}")
            st.write(f"**AI Response:** {item.get('ai_response', 'No AI response available')}")
            if item.get('priority') == 'High':
                st.warning("⚠️ High Priority")


def show_search():
    st.title("🧠 Smart Email Search")
    if not st.session_state.get('rag_pipeline'):
        st.warning("Please build the Smart Search index from the Dashboard page before searching.")
        if st.button("Go to Dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()
        return

    query = st.text_input("🔍 Enter your question:", placeholder="e.g., What was the revenue growth in Q3?")

    if query:
        with st.spinner("Searching through your emails..."):
            rag_pipeline = st.session_state['rag_pipeline']
            answer, sources = rag_pipeline.answer(query)

            st.markdown("### ✅ Answer")
            st.markdown(f"<div class='rag-answer'>{html.escape(answer)}</div>", unsafe_allow_html=True)

            st.markdown("### 📚 Sources Found")
            if sources:
                for i, source in enumerate(sources):
                    entity = source.get('entity', {})
                    with st.container():
                        st.markdown(f"<div class='rag-source'>", unsafe_allow_html=True)
                        st.markdown(f"**Source [{i + 1}]** | Similarity Distance: {source.get('distance'):.4f}")
                        st.markdown(f"**From:** {html.escape(entity.get('sender', 'N/A'))}")
                        st.markdown(f"**Subject:** {html.escape(entity.get('subject', 'N/A'))}")
                        with st.expander("View full context"):
                            st.text(entity.get('context', 'No context available.'))
                        st.markdown(f"</div>", unsafe_allow_html=True)
            else:
                st.info("No specific source emails were used to generate the answer.")


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(image_file="2.jpg"):
    bin_str = get_base64_of_bin_file(image_file)
    st.markdown(
        f"""
        body {{
            background: linear-gradient(rgba(30,30,30,0.7), rgba(30,30,30,0.7)),
                        url("data:image/jpg;base64,{bin_str}") no-repeat center center fixed !important;
            background-size: cover !important;
            background-attachment: fixed !important;
        }}
        .stApp {{
            background: transparent !important;
        }}
        .block-container {{
            background: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def show_settings():
    st.title("Settings")
    bg_options = ["None", "2.jpg", "Photo1.jpg"]
    bg_choice = st.selectbox(
        "Background Image",
        bg_options,
        index=bg_options.index(st.session_state.get('bg_image', "None")) if st.session_state.get(
            'bg_image') in bg_options else 0
    )
    if bg_choice != "None":
        st.image(bg_choice, caption="Background Preview", use_container_width=True)
    if st.button("Apply Settings"):
        if bg_choice == "None":
            st.session_state['bg_image'] = None
        else:
            st.session_state['bg_image'] = bg_choice
        st.rerun()
        return
    st.subheader("User Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Name", value=st.session_state.user.name, disabled=True)
    with col2:
        st.text_input("Email", value=st.session_state.user.email, disabled=True)
    st.subheader("Email Preferences")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.emails_per_page = st.number_input(
            "Emails per page",
            min_value=10,
            max_value=50,
            value=st.session_state.emails_per_page
        )
    with col2:
        st.session_state.default_sort = st.selectbox(
            "Default Sort Order",
            ["Newest First", "Oldest First"]
        )
    st.subheader("Notification Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.notify_high_priority = st.toggle(
            "Notify for High Priority Emails",
            value=st.session_state.get('notify_high_priority', True)
        )
    with col2:
        st.session_state.notify_action_items = st.toggle(
            "Notify for Action Items",
            value=st.session_state.get('notify_action_items', True)
        )
    st.subheader("Theme Settings")
    if 'theme' not in st.session_state:
        st.session_state.theme = "Light"
    selected_theme = st.selectbox(
        "Theme",
        ["Light", "Dark", "System"],
        key="theme_selector"
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()
        return
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")


def logout():
    st.cache_data.clear()
    st.cache_resource.clear()
    for key in list(st.session_state.keys()):
        if key not in ['page']:
            del st.session_state[key]
    st.session_state.page = 'login'


def show_todo():
    st.header("📋 To-Do List")
    if not st.session_state.email_processor:
        st.warning("Please connect your Gmail account first")
        return

    user = st.session_state.user
    todos = st.session_state.email_processor.get_todo_emails(user=user)
    if not todos:
        st.info("No to-do emails found.")
        return

    from collections import defaultdict
    todos_by_date = defaultdict(list)
    for email in todos:
        date_str = email['date']
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        todos_by_date[date_obj].append(email)

    for date, emails in sorted(todos_by_date.items(), reverse=True):
        st.subheader(f"{date.strftime('%B %d, %Y')}")
        for email in emails:
            with st.container():
                col1, col2 = st.columns([0.8, 0.2])
                subject = email.get('subject', '📧 No Subject')
                sender = email.get('sender', 'Unknown')
                summary = email.get('summary', '').split('.')[0]
                ai_response = email.get('ai_response', '').split('.')[0] if email.get('ai_response') else ''
                with col1:
                    task = f"• **{subject}**\n"
                    task += f"  - From: {sender}\n"
                    task += f"  - Priority Task: {summary if summary else 'No summary available'}\n"
                    if ai_response:
                        task += f"  - AI Suggestion: {ai_response}\n"
                    st.markdown(task)
                with col2:
                    if st.button("📧 View Full", key=f"view_{email.get('id', subject)}"):
                        with st.expander("Full Email Details", expanded=True):
                            st.write(f"**Subject:** {subject}")
                            st.write(f"**From:** {sender}")
                            st.write(f"**Time:** {email.get('timestamp', 'Unknown')}")
                            st.write("**Content:**")
                            st.write(email.get('body', 'No content available'))
                            if email.get('summary'):
                                st.info(f"**AI Summary:** {email['summary']}")
                            if email.get('ai_response'):
                                st.success(f"**AI Response:** {email['ai_response']}")
            st.divider()


def main():
    apply_custom_css()
    apply_theme()
    st.title("MailMentor")
    if st.session_state.user is None:
        if st.session_state.page == 'register':
            show_register()
        else:
            show_login()
        return

    session = SessionLocal()
    try:
        user = session.merge(st.session_state.user)
        with st.sidebar:
            st.title("Navigation")
            st.write(f"Welcome, {user.name}!")
            st.button("Dashboard", on_click=lambda: st.session_state.update(page="Dashboard"))
            st.button("Inbox", on_click=lambda: st.session_state.update(page="Inbox"))
            st.button("Action Items", on_click=lambda: st.session_state.update(page="Action Items"))
            st.button("Smart Search", on_click=lambda: st.session_state.update(page="Smart Search"))
            st.button("Settings", on_click=lambda: st.session_state.update(page="Settings"))
            st.button("To-Do", on_click=lambda: st.session_state.update(page="To-Do"))
            if st.button("Logout", on_click=logout):
                st.rerun()

        if st.session_state.page == "Dashboard":
            show_dashboard()
        elif st.session_state.page == "Inbox":
            show_inbox()
        elif st.session_state.page == "Action Items":
            show_action_items()
        elif st.session_state.page == "Smart Search":
            show_search()
        elif st.session_state.page == "Settings":
            show_settings()
        elif st.session_state.page == "To-Do":
            show_todo()
    finally:
        session.close()


if __name__ == "__main__":
    main()

# Important Persons Configuration
IMPORTANT_DOMAINS = [
    '@company.com',
    '@internal.org',
    '@client.com',
    '@partner.com'
]

IMPORTANT_EMAILS = [
    'ceo@company.com',
    'manager@company.com',
]

IMPORTANT_ROLES = [
    'director', 'manager', 'lead', 'head',
    'ceo', 'cto', 'cfo', 'vp', 'president'
]

