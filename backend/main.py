import os
import shutil
import json
import datetime
import importlib
from typing import Dict, Any

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from scripts import pdf_summarizer, pdf_qa

app = FastAPI(title="Automation Dashboard Backend")

# --- Pydantic Models ---
class WebscraperRequest(BaseModel):
    url: str = "https://www.theverge.com"
    params: Dict[str, Any] = {}

class AskQARequest(BaseModel):
    question: str
# -------------------------------------
# CORS (React compatibility)
# -------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load config (optional)
if os.path.exists("config/config.json"):
    with open("config/config.json", "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {}

if not os.path.exists("outputs"):
    os.makedirs("outputs")

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# -------------------------------------
# BASIC ROOT ENDPOINT
# -------------------------------------
@app.get("/")
def read_root():
    return {"message": "Automation Dashboard Backend Running"}

# -------------------------------------
# WEBSCRAPER ENDPOINT
# -------------------------------------
@app.post("/run/webscraper")
def run_script(body: WebscraperRequest):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("logs", exist_ok=True)

    try:
        script_module = importlib.import_module("scripts.webscraper")
        params = body.dict()

        result = script_module.run(params, timestamp)

        log_file = f"logs/webscraper_{timestamp}.log"
        with open(log_file, "w") as log:
            log.write(
                f"webscraper ran successfully at {timestamp}\n"
                f"{json.dumps(result, indent=2)}"
            )

        return {"status": "success", "log": log_file, "output": result}

    except Exception as e:
        log_file = f"logs/webscraper_{timestamp}.log"
        with open(log_file, "w") as log:
            log.write(f"webscraper failed at {timestamp}\n{str(e)}")
        return {"status": "failed", "log": log_file, "error": str(e)}


# -------------------------------------
# DOWNLOAD LATEST CSV
# -------------------------------------
@app.get("/download-latest-csv")
def download_latest_csv():
    base_dir = "outputs/webscraper"
    if not os.path.exists(base_dir):
        return {"error": "No CSV found yet"}

    latest_folder = sorted(os.listdir(base_dir))[-1]
    csv_path = os.path.join(base_dir, latest_folder, "articles.csv")

    if os.path.exists(csv_path):
        return FileResponse(csv_path, media_type="text/csv", filename="articles.csv")

    return {"error": "CSV not found in latest output"}


import hashlib

# ... (other imports)

# ---------------------------------------------------
# PDF Summarizer
# ---------------------------------------------------
PROCESSED_FILES_DB = os.path.join(UPLOAD_DIR, "processed_files.json")

def get_processed_files():
    if not os.path.exists(PROCESSED_FILES_DB):
        return {}
    with open(PROCESSED_FILES_DB, "r") as f:
        return json.load(f)

def add_processed_file(file_hash, filename, timestamp):
    db = get_processed_files()
    db[file_hash] = {"filename": filename, "timestamp": timestamp}
    with open(PROCESSED_FILES_DB, "w") as f:
        json.dump(db, f, indent=2)

@app.post("/run/pdf_summarizer")
def run_pdf_summarizer(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Read file content and calculate hash
    content = file.file.read()
    file_hash = hashlib.sha256(content).hexdigest()

    # Check for duplicates
    processed_files = get_processed_files()
    if file_hash in processed_files:
        original = processed_files[file_hash]
        return {
            "status": "duplicate",
            "message": f"This file is a duplicate of '{original['filename']}' processed at {original['timestamp']}.",
            "original_filename": original['filename'],
            "original_timestamp": original['timestamp'],
        }

    # Save PDF
    temp_path = os.path.join("uploads", f"{timestamp}_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(content)

    # Add to processed files DB
    add_processed_file(file_hash, file.filename, timestamp)

    try:
        result = pdf_summarizer.run({"file_path": temp_path}, timestamp)

        return {
            "status": "success",
            "summary_file": result.get("summary_file"),
            "extracted_file": result.get("extracted_file"),
            "vector_db": result.get("vector_db"),
            "chunks_added": result.get("count_chunks"),
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}


# ---------------------------------------------------
# PDF Question Answering
# ---------------------------------------------------
@app.post("/run/ask_qa")
def ask_qa(body: AskQARequest):
    question = body.question.strip()

    if not question:
        return {"status": "failed", "error": "No question provided"}

    try:
        # pdf_qa.run returns full dict with status, answer, sources, context
        result = pdf_qa.run({"question": question, "top_k": 3})
        return result

    except Exception as e:
        return {"status": "failed", "error": str(e)}