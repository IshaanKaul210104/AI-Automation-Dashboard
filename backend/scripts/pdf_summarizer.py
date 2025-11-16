# backend/scripts/pdf_summarizer.py
import os
import glob
import json
import datetime
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import ollama

# ----------------------
# Config - loaded from file
# ----------------------
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["pdf_summarizer"]

config = load_config()

# Expand HF home path
HF_HOME = os.path.expanduser(config["hf_home"])
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Directly use config values (NO nested pdf_summarizer lookup)
BLIP_ROOT = os.path.join(HF_HOME, config["blip_root"])
EMBED_MODEL_NAME = config["embed_model_name"]

# vector DB location
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "vector_db")
VECTOR_DIR = os.path.abspath(VECTOR_DIR)
os.makedirs(VECTOR_DIR, exist_ok=True)
INDEX_PATH = os.path.join(VECTOR_DIR, "pdf_index.faiss")
CHUNKS_PATH = os.path.join(VECTOR_DIR, "chunks.json")

# ----------------------
# Load BLIP captioning model
# ----------------------
print("--- Loading BLIP captioning model ---")

def find_snapshot_folder(model_root):
    snaps = os.path.join(model_root, "snapshots")
    if not os.path.isdir(snaps):
        raise RuntimeError(f"No snapshots folder in {model_root}")
    subfolders = [os.path.join(snaps, d) for d in os.listdir(snaps)]
    subfolders = [p for p in subfolders if os.path.isdir(p)]
    if not subfolders:
        raise RuntimeError("Snapshots folder is empty.")
    return subfolders[0]

BLIP_SNAPSHOT = find_snapshot_folder(BLIP_ROOT)
print(f"Using BLIP snapshot: {BLIP_SNAPSHOT}")

blip_processor = BlipProcessor.from_pretrained(BLIP_SNAPSHOT, local_files_only=True)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_SNAPSHOT, local_files_only=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model.to(device)
blip_model.eval()

print("--- BLIP model loaded successfully ---")

# ----------------------
# Load sentence-transformers model (for embeddings)
# ----------------------
print("--- Loading embedding model ---")
embedder = SentenceTransformer(EMBED_MODEL_NAME)
print("--- Embedding model loaded successfully ---")

# ----------------------
# PDF extraction + image captioning
# ----------------------
def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    content = []
    for page in doc:
        text = page.get_text()
        if text:
            content.append({"type": "text", "content": text})
        
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                img_bytes = base.get("image")
                if img_bytes:
                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    content.append({"type": "image", "content": img})
            except Exception:
                continue
    return content

def caption_images(content):
    # replace image objects with caption strings
    for i, item in enumerate(content):
        if item.get("type") == "image":
            img = item["content"]
            try:
                inputs = blip_processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = blip_model.generate(**inputs, max_new_tokens=40)

                caption = blip_processor.decode(out[0], skip_special_tokens=True)
                content[i] = {"type": "image", "content": caption}
            except Exception as e:
                content[i] = {"type": "image", "content": f"[caption failed: {e}]"}
    return content

def create_ordered_text(content):
    parts = []
    for item in content:
        if item["type"] == "text":
            parts.append(item["content"])
        else:
            parts.append(f"[Image]: {item['content']}")
    return "\n\n".join(parts)

# ----------------------
# Chunking helper (simple char-based chunking with overlap)
# ----------------------
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

# ----------------------
# FAISS utilities: append embeddings and chunks
# ----------------------
def l2_normalize(x: np.ndarray):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return x / norms

def save_chunks_and_index(all_chunks_texts, all_meta, embeddings):
    """
    all_chunks_texts: list[str] (new chunks)
    all_meta: list[dict] (metadata dicts for each chunk: filename, start, end, timestamp)
    embeddings: np.ndarray float32 (num_new x dim)
    This function will append to existing index/chunks or create new ones.
    """
    dim = embeddings.shape[1]
    # normalize for inner-product (cosine)
    embeddings = l2_normalize(embeddings).astype("float32")

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        # check dim
        if index.d != dim:
            raise RuntimeError(f"Existing index dim {index.d} != new dim {dim}")
        # append
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH)
        # append chunks metadata
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.extend(all_meta)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    else:
        # create new IndexFlatIP (inner product on normalized vectors => cosine)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_meta, f, ensure_ascii=False, indent=2)

# ----------------------
# Ollama summarizer util (via CLI)
# ----------------------
import ollama

# ... (imports)

# ----------------------
# Ollama summarizer util (via python client)
# ----------------------
def summarize_with_ollama(prompt_text, model="granite3.2:8b", timeout=300):
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            options={"temperature": 0.0},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[Ollama failed: {e}]"

# ----------------------
# Main run() entrypoint
# ----------------------
def run(params=None, timestamp=None):
    """
    params: dict expected to contain:
      - file_path: path to saved PDF
      - uploader (optional): user/file metadata
      - ollama_model (optional)
    """
    params = params or {}
    pdf_path = params.get("file_path")
    timestamp = timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", "pdf_summaries", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "log.txt")

    try:
        if not pdf_path or not os.path.exists(pdf_path):
            raise FileNotFoundError("PDF path missing or not found.")

        # 1) extract
        content = extract_text_and_images(pdf_path)

        # 2) caption images
        content = caption_images(content)

        # 3) merge ordered text
        ordered_text = create_ordered_text(content)

        # 4) save basic outputs
        summary_prompt = f"Summarize the following PDF content in order. Include image descriptions inline:\n\n{ordered_text[:6000]}"
        summary = summarize_with_ollama(summary_prompt, model=params.get("ollama_model", "granite3.2:8b"))

        summary_path = os.path.join(output_dir, "combined_summary.txt")
        extracted_path = os.path.join(output_dir, "extracted_text.txt")
        json_path = os.path.join(output_dir, "content.json")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        with open(extracted_path, "w", encoding="utf-8") as f:
            f.write(ordered_text)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"content": content}, f, ensure_ascii=False, indent=2)

        # 5) chunk & embed & store into FAISS (append)
        chunks = chunk_text(ordered_text, chunk_size=1200, overlap=200)
        metas = []
        for i, c in enumerate(chunks):
            metas.append({
                "text": c,
                "source_file": os.path.basename(pdf_path),
                "chunk_index": i,
                "timestamp": timestamp
            })
        # embed
        embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        # save to vector DB
        save_chunks_and_index(chunks, metas, embeddings)

        # log
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Processed: {pdf_path}\nSummary: {summary_path}\nExtracted: {extracted_path}\nVector DB: {VECTOR_DIR}\n")

        return {
            "status": "success",
            "summary_file": summary_path,
            "extracted_file": extracted_path,
            "content_json": json_path,
            "vector_db": VECTOR_DIR,
            "count_chunks": len(chunks),
            "log": log_file
        }

    except Exception as e:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Failed: {str(e)}\n")
        return {"status": "failed", "error": str(e), "log": log_file}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        result = run({"file_path": pdf_path})
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python pdf_summarizer.py <path_to_pdf>")