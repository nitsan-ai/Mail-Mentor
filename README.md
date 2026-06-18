# MailMentor

**An AI-powered Gmail assistant** that turns your inbox into an actionable dashboard. Connect your Google account, sync emails to PostgreSQL, and use local NLP plus RAG (Retrieval-Augmented Generation) to summarize messages, surface action items, and search your mail in natural language.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

| Area | What it does |
|------|----------------|
| **Authentication** | Register and log in with email/password; Gmail is linked via Google OAuth 2.0 |
| **Dashboard** | Inbox metrics, category charts, recent emails, and one-click CSV export |
| **AI Summaries** | On-demand BART-based summarization and suggested replies per email |
| **Smart Search (RAG)** | Ask natural-language questions; answers are grounded in your synced emails |
| **Inbox & Action Items** | Browse synced mail and filter categorized action items by status |
| **To-Do View** | Emails marked as to-do, grouped by date with AI hints |
| **Email Categorization** | Rule-based tagging (Urgent, Meeting, Invoice, Follow-up, Important, etc.) |
| **Per-user isolation** | Emails and vector indexes are scoped to the logged-in user |
| **Settings** | Theme, pagination, notifications, and optional background images |

---

## Architecture

```mermaid
flowchart LR
    subgraph Client
        UI[Streamlit UI]
    end

    subgraph Auth
        Login[Email / Password Auth]
        OAuth[Google OAuth 2.0]
    end

    subgraph Data
        PG[(PostgreSQL)]
        Milvus[(Milvus Lite)]
    end

    subgraph AI
        BART[BART Summarizer]
        ST[Sentence Transformers]
        Groq[Groq LLM]
    end

    Gmail[Gmail API]

    UI --> Login
    UI --> OAuth
    OAuth --> Gmail
    Gmail --> PG
    UI --> PG
    PG --> ST
    ST --> Milvus
    UI --> Milvus
    Milvus --> Groq
    UI --> BART
```

**Data flow (simplified)**

1. User registers or logs in → Gmail OAuth stores credentials on the user record.
2. `EmailProcessor` fetches mail via the Gmail API and persists it in PostgreSQL.
3. On login or refresh, emails are embedded with `sentence-transformers/all-MiniLM-L6-v2` and indexed in a per-user Milvus Lite collection.
4. **Smart Search** retrieves relevant chunks and sends them to Groq (Llama 3) for grounded answers.
5. **AI Analysis** uses a local Hugging Face BART model for summaries (no API call required).

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | [Streamlit](https://streamlit.io/) |
| Database | PostgreSQL + [SQLAlchemy](https://www.sqlalchemy.org/) |
| Vector store | [Milvus Lite](https://milvus.io/docs/milvus_lite.md) (`pymilvus`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via LlamaIndex |
| RAG LLM | [Groq API](https://console.groq.com/) (Llama 3 8B) |
| Summarization | `facebook/bart-large-cnn` (local, via Transformers) |
| Email sync | Gmail API (`google-api-python-client`) |
| Auth | Werkzeug password hashing + Google OAuth 2.0 |

---

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **PostgreSQL 14+**
- **Google Cloud project** with Gmail API enabled
- **Groq API key** (for Smart Search only)
- ~4 GB free disk space (for Hugging Face model downloads on first run)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://gitlab.nitsantech.com/nitsan/ai/mail-mentor.git
cd mail-mentor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install pymilvus pandas altair "llama-index-llms-groq" "llama-index-embeddings-huggingface"
```

> **Note:** `pymilvus`, `pandas`, and `altair` are required at runtime but may not be listed in `requirements.txt`. Install them explicitly if you hit import errors.

### 4. Set up PostgreSQL

**macOS (Homebrew)**

```bash
brew install postgresql@16
brew services start postgresql@16
psql postgres
```

**Ubuntu / Debian**

```bash
sudo apt update && sudo apt install postgresql postgresql-contrib
sudo service postgresql start
sudo -u postgres psql
```

**Create database and user**

```sql
CREATE USER mailmentor_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE mailmentor_db OWNER mailmentor_user;
GRANT ALL PRIVILEGES ON DATABASE mailmentor_db TO mailmentor_user;
\q
```

**Verify the connection**

```bash
psql -U mailmentor_user -d mailmentor_db -h localhost
```

Tables (`users`, `emails`) are created automatically by SQLAlchemy on first run.

### 5. Configure Google OAuth credentials

1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project (or select an existing one).
3. Enable the **Gmail API** (`APIs & Services → Library`).
4. Configure the **OAuth consent screen** (External or Internal).
5. Create credentials → **OAuth client ID** → **Desktop app**.
6. Download the JSON file and save it as:

   ```
   config/credentials.json
   ```

   The app reads this path in `main.py` when connecting Gmail.

### 6. Configure environment variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://mailmentor_user:your_secure_password@localhost:5432/mailmentor_db
GROQ_API_KEY=your_groq_api_key_here
```

Get a Groq API key at [console.groq.com](https://console.groq.com/).

### 7. Run the application

```bash
streamlit run main.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

---

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `GROQ_API_KEY` | Yes (for Smart Search) | Groq API key for RAG answers |

**Google credentials** live at `config/credentials.json` (not in `.env`).

**Optional UI assets** — background images for Settings are in `assets/img/` (`2.jpg`, `Photo1.jpg`).

**Model settings** — summarization and local model options are defined in `app/config.py`.

---

## Usage

### First-time setup

1. **Register** with your name, email, and password.
2. Complete the **Google OAuth** flow in the browser when prompted.
3. Wait while the app builds your **private Smart Search index** (embeddings + Milvus).
4. You land on the **Dashboard**.

### Navigation

| Page | Description |
|------|-------------|
| **Dashboard** | Metrics, charts, recent mail, AI analysis tab, refresh, CSV export |
| **Inbox** | Full list of synced emails |
| **Action Items** | Emails tagged as action items; filter by Pending / Completed |
| **Smart Search** | Natural-language Q&A over your indexed emails |
| **To-Do** | To-do emails grouped by date |
| **Settings** | Theme, pagination, notifications, background image |

### Smart Search tips

- Build or rebuild the index from the Dashboard (happens automatically on login/refresh).
- Ask specific questions, e.g. *"What did John say about the Q3 budget?"*
- Answers cite source emails with similarity scores.

### Refreshing emails

Use **Refresh** on the Dashboard to pull the latest messages from Gmail and rebuild the search index.

---

## Project Structure

```
mail-mentor/
├── main.py                  # Streamlit entry point and UI
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create locally; do not commit)
├── config/
│   └── credentials.json     # Google OAuth client secrets (do not commit)
├── assets/
│   └── img/                 # Optional background images
└── app/
    ├── __init__.py
    ├── ai_services.py       # BART summarization and AI analysis
    ├── auth.py              # User authentication helpers
    ├── categorization.py    # Rule-based email categorization
    ├── config.py            # Database URL and model settings
    ├── email_processor.py   # Gmail API sync, stats, action items, to-dos
    ├── models.py            # SQLAlchemy models (User, Email)
    ├── rag.py               # RAG pipeline (embeddings, Milvus, Groq)
    ├── utils.py             # Email parsing and query helpers
    └── vector_store.py      # Milvus vector store utilities
```

---

## Security

- **Never commit** `.env`, `config/credentials.json`, or any file containing API keys or OAuth secrets.
- Gmail credentials are stored encrypted as JSON on the user record after OAuth completes.
- Each user's emails and Milvus collection are isolated by `user_id`.
- Use strong passwords and restrict PostgreSQL access to trusted hosts.
- For production, run behind HTTPS, use a secrets manager, and rotate API keys regularly.

---

## Troubleshooting

| Issue | Likely cause | Fix |
|-------|--------------|-----|
| `Database connection failed` | Wrong `DATABASE_URL` or PostgreSQL not running | Verify connection with `psql`; check `.env` |
| `Failed to connect to Gmail` | Missing or invalid `config/credentials.json` | Re-download OAuth credentials; ensure Gmail API is enabled |
| Smart Search returns errors | Missing `GROQ_API_KEY` | Add key to `.env` and restart the app |
| `ModuleNotFoundError: pymilvus` | Package not installed | `pip install pymilvus` |
| Slow first startup | Models downloading from Hugging Face | Normal; subsequent runs are faster |
| Empty dashboard after login | No emails synced yet | Click **Refresh** on the Dashboard |
| OAuth redirect issues | Desktop app type not used | Recreate credentials as **Desktop app** in Google Cloud |

---

## License

Copyright © [NITSAN Technologies LLP](https://nitsantech.com/). All rights reserved.
