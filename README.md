# AI Automation Dashboard

A full-stack web application that provides a unified interface for running various AI/ML automation tasks. It consists of a **React** frontend and a **Python (FastAPI)** backend, communicating over a REST API.

## Features

- **Web Scraper** — Scrape articles from any website
- **PDF Summarizer** — Summarize PDF documents with image captioning
- **PDF Q&A** — Ask questions about uploaded PDFs (RAG-based)
- **ML Model Recommender** — Get model recommendations for a given dataset
- **Audio Transcriber** — Transcribe audio files to text

---

## Architecture

```
frontend/ (React + Vite)    <--->    backend/ (FastAPI + Uvicorn)
     :5173 (dev)                          :8000
     Axios HTTP calls       REST API    Python scripts
```

**Frontend:** React 19, Vite 5, Tailwind CSS, Axios, Lucide React icons
**Backend:** FastAPI, Uvicorn, Ollama (local LLM), FAISS vector DB, Sentence Transformers, PyMuPDF, BeautifulSoup, Whisper, XGBoost

---

## Features & Modules

### Web Scraper (`/run/webscraper`)
- Scrapes article titles and links from any URL
- Uses BeautifulSoup + lxml for parsing
- Outputs results as JSON and CSV files
- Supports downloading the latest CSV via `/download-latest-csv`

### PDF Summarizer (`/run/pdf_summarizer`)
- Extracts text and images from uploaded PDFs using PyMuPDF (fitz)
- Captions images using Salesforce BLIP model
- Generates summaries via a local Ollama LLM (`granite3.2:8b`)
- Stores document chunks in a FAISS vector database for later Q&A
- Detects duplicate uploads using SHA-256 hashing

### PDF Question Answering (`/run/ask_qa`)
- RAG (Retrieval-Augmented Generation) pipeline
- Encodes questions with sentence-transformers (`all-MiniLM-L6-v2`)
- Retrieves top-k similar chunks from FAISS index
- Generates answers using local Ollama LLM with retrieved context

### ML Model Recommender (`/run/model_recommender`)
- Accepts a CSV dataset with a specified task (regression/classification/clustering)
- Computes meta-features (skewness, correlation, multicollinearity)
- Recommends the best model based on data characteristics
- Supports: LinearRegression, Ridge, RandomForest, XGBoost, LogisticRegression, KMeans, DBSCAN
- Evaluates model with cross-validation and returns metrics

### Audio Transcriber (`/run/transcribe_audio`)
- Accepts audio file uploads
- Transcribes speech to text using OpenAI Whisper (base model)
- Returns a downloadable `.txt` transcript

---

## Libraries Used

### Frontend (Node.js / npm)

**Production:**
| Library | Version | Description |
|---------|---------|-------------|
| react | ^19.1.1 | UI library |
| react-dom | ^19.1.1 | React DOM rendering |
| axios | ^1.12.2 | HTTP client for API calls |
| lucide-react | ^0.577.0 | Icon library |

**Dev Dependencies:**
| Library | Version | Description |
|---------|---------|-------------|
| vite | ^5.3.1 | Build tool and dev server |
| @vitejs/plugin-react | ^5.0.4 | Vite plugin for React |
| tailwindcss | ^3.4.13 | Utility-first CSS framework |
| postcss | ^8.5.6 | CSS post-processing |
| autoprefixer | ^10.4.21 | CSS vendor prefixing |
| eslint | ^9.36.0 | Linter |
| eslint-plugin-react-hooks | ^5.2.0 | ESLint rules for React hooks |
| eslint-plugin-react-refresh | ^0.4.22 | ESLint rules for React Refresh |

### Backend (Python)

**Core Framework:**
| Library | Version | Description |
|---------|---------|-------------|
| fastapi | 0.119.1 | Web framework for building APIs |
| uvicorn | 0.38.0 | ASGI server |
| pydantic | 2.12.3 | Data validation and serialization |
| python-multipart | 0.0.20 | File upload support |

**Web Scraping:**
| Library | Version | Description |
|---------|---------|-------------|
| beautifulsoup4 | 4.14.2 | HTML/XML parsing |
| lxml | 6.0.2 | Fast XML/HTML parser |
| requests | 2.32.5 | HTTP library |

**PDF Processing:**
| Library | Version | Description |
|---------|---------|-------------|
| PyMuPDF | 1.26.5 | PDF text and image extraction (fitz) |

**AI / ML:**
| Library | Version | Description |
|---------|---------|-------------|
| torch | 2.9.0 | PyTorch deep learning framework |
| transformers | 4.57.1 | Hugging Face Transformers (BLIP model) |
| sentence-transformers | 5.1.2 | Sentence embeddings (all-MiniLM-L6-v2) |
| ollama | 0.6.0 | Python client for Ollama local LLM |
| faiss-cpu | 1.12.0 | Facebook AI Similarity Search (vector DB) |
| scikit-learn | 1.7.2 | ML algorithms and evaluation |
| xgboost | — | Gradient boosting framework |
| openai-whisper | — | Speech-to-text (installed via git) |

**Data Processing:**
| Library | Version | Description |
|---------|---------|-------------|
| numpy | 2.3.4 | Numerical computing |
| pandas | — | Data manipulation and analysis |
| scipy | 1.16.3 | Scientific computing |
| pillow | 12.0.0 | Image processing |

**Other:**
| Library | Version | Description |
|---------|---------|-------------|
| huggingface-hub | 0.36.0 | Hugging Face model hub utilities |
| httpx | 0.28.1 | Async HTTP client |
| PyYAML | 6.0.3 | YAML parsing |
| tqdm | 4.67.1 | Progress bars |

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- **Ollama** installed and running locally — [https://ollama.com](https://ollama.com)
- Ollama model pulled:
  ```bash
  ollama pull granite3.2:8b
  ```
- Hugging Face models cached locally:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `Salesforce/blip-image-captioning-large`

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd AI-Automation-Dashboard
```

### Step 2: Start the Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install Python dependencies
pip install -r requirements.txt

# Install Whisper separately (for audio transcription)
python -m pip install git+https://github.com/openai/whisper.git

# Make sure Ollama is running
ollama serve                      # In a separate terminal if needed

# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Backend API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger UI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Step 3: Start the Frontend

```bash
cd frontend

# Install Node.js dependencies
npm install

# Start the Vite dev server
npm run dev
```

- Frontend: [http://localhost:5173](http://localhost:5173)

### Step 4: Use the Dashboard

Open [http://localhost:5173](http://localhost:5173) in your browser. The dashboard presents a 3-column layout:

| Left Column | Center Column | Right Column |
|-------------|---------------|--------------|
| Web Scraper | PDF Summarizer | ML Model Recommender |
| Audio Transcriber | PDF Q&A | |

### Optional: Build for Production

```bash
cd frontend
npm run build
# Output will be in frontend/dist/
```

---

## Project Structure

```
AI-Automation-Dashboard/
├── README.md
├── PROJECT_SUMMARY.txt
├── backend/
│   ├── main.py                    # FastAPI app with all API endpoints
│   ├── requirements.txt           # Python dependencies
│   ├── config/
│   │   └── config.json            # Scraper and model configuration
│   ├── scripts/
│   │   ├── webscraper.py          # Web scraping logic
│   │   ├── pdf_summarizer.py      # PDF extraction, captioning, summarization
│   │   ├── pdf_qa.py              # RAG-based PDF question answering
│   │   ├── model_recommender.py   # ML model recommendation engine
│   │   └── audio_transcriber.py   # Whisper-based audio transcription
│   ├── uploads/                   # Temporary uploaded files
│   ├── outputs/                   # Script output files (CSV, JSON)
│   ├── logs/                      # Execution logs
│   ├── vector_db/                 # FAISS index and chunk metadata
│   │   ├── pdf_index.faiss
│   │   └── chunks.json
│   └── venv/                      # Python virtual environment
└── frontend/
    ├── package.json               # Node.js dependencies and scripts
    ├── vite.config.js             # Vite configuration
    ├── index.html                 # HTML entry point
    ├── public/                    # Static assets
    └── src/
        ├── main.jsx               # React entry point
        ├── App.jsx                # Root App component
        ├── index.css              # Global styles (Tailwind)
        ├── api/
        │   └── api.js             # Axios API client
        └── components/
            ├── Dashboard.jsx          # Main dashboard layout
            ├── WebScraperButton.jsx   # Web scraper UI
            ├── PDFSummarizerButton.jsx# PDF summarizer UI
            ├── AskQAButton.jsx        # PDF Q&A UI
            ├── MLModelRecommender.jsx # Model recommender UI
            ├── AudioTranscriberButton.jsx # Audio transcriber UI
            └── RunButton.jsx          # Reusable run button component
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/run/webscraper` | Run web scraper (body: `{url, params}`) |
| `GET` | `/download-latest-csv` | Download latest scraped CSV |
| `POST` | `/run/pdf_summarizer` | Upload & summarize a PDF (multipart file) |
| `POST` | `/run/ask_qa` | Ask a question about uploaded PDFs (body: `{question}`) |
| `POST` | `/run/model_recommender` | Upload CSV for model recommendation (multipart file + query: `task`, `target_col`) |
| `POST` | `/run/transcribe_audio` | Upload audio for transcription (multipart file) |
