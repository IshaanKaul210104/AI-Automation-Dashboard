from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json, os, datetime, importlib
from fastapi.staticfiles import StaticFiles
from scripts import pdf_summarizer

app = FastAPI(title="Automation Dashboard API")

# Enable CORS (for React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config (optional)
if os.path.exists("config/config.json"):
    with open("config/config.json", "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {}

if not os.path.exists("outputs"):
    os.makedirs("outputs")

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/")
def read_root():
    return {"message": "Automation Dashboard Backend Running"}

@app.post("/run/webscraper")
async def run_script(request: Request):
    """
    Dynamically runs scripts like 'webscraper.py' under /scripts folder.
    Accepts optional JSON body (e.g., {'url': 'https://example.com'}).
    """
    body = await request.json()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("logs", exist_ok=True)

    try:
        try:
            script_module = importlib.import_module(f"scripts.webscraper")
        except ModuleNotFoundError:
            raise

        # ✅ Ensure URL is valid or fallback to default
        url = body.get("url", "").strip() or "https://www.theverge.com"
        params = body if isinstance(body, dict) else {}

        # Run script (with both URL and timestamp for flexibility)
        if hasattr(script_module, "run"):
            result = script_module.run(params, timestamp)
        elif hasattr(script_module, "run_web_scraper"):
            result = script_module.run_web_scraper(url)
        else:
            raise Exception(f"No valid entry point found in webscraper.py")

        # Write logs
        log_file = f"logs/webscraper_{timestamp}.log"
        with open(log_file, "w") as log:
            log.write(f"webscraper ran successfully at {timestamp}\n{json.dumps(result, indent=2)}")

        return {"status": "success", "log": log_file, "output": result}

    except Exception as e:
        log_file = f"logs/webscraper_{timestamp}.log"
        with open(log_file, "w") as log:
            log.write(f"webscraper failed at {timestamp}\n{str(e)}")
        return {"status": "failed", "log": log_file, "error": str(e)}
    
@app.get("/download-latest-csv")
def download_latest_csv():
    base_dir = "outputs/webscraper"
    if not os.path.exists(base_dir):
        return {"error": "No CSV found yet"}
    
    # Find the most recent timestamped folder
    latest_folder = sorted(os.listdir(base_dir))[-1]
    csv_path = os.path.join(base_dir, latest_folder, "articles.csv")
    
    if os.path.exists(csv_path):
        return FileResponse(csv_path, media_type="text/csv", filename="articles.csv")
    else:
        return {"error": "CSV not found in latest output"}