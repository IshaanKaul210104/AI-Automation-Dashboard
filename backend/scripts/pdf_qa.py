# backend/scripts/pdf_qa.py

import os
import json
import numpy as np
import faiss
import subprocess
import ollama
from sentence_transformers import SentenceTransformer

# -----------------------
# CONFIG
# -----------------------
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["pdf_summarizer"]

config = load_config()

HF_HOME = os.path.expanduser(config["hf_home"])
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_OFFLINE"] = "1"

VECTOR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vector_db"))
INDEX_PATH = os.path.join(VECTOR_DIR, "pdf_index.faiss")
CHUNKS_PATH = os.path.join(VECTOR_DIR, "chunks.json")

EMBED_MODEL_NAME = config["embed_model_name"]
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# -----------------------
# HELPERS
# -----------------------
def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return x / norms


import ollama

def ollama_answer(question, context, model="granite3.2:8b"):
    prompt = f"""
You are a helpful assistant. Use the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[Ollama failed: {e}]"


def load_vector_store():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("FAISS index missing. Upload + summarize a PDF first.")
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("chunks.json missing. Upload + summarize a PDF first.")

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


# -----------------------
# MAIN ENTRYPOINT: run()
# -----------------------
def run(params=None, timestamp=None):
    params = params or {}
    question = params.get("question", "").strip()
    top_k = int(params.get("top_k", 3))

    if not question:
        return {"status": "failed", "error": "Question is required."}

    try:
        index, chunks = load_vector_store()

        # Encode and normalize question
        q_emb = embedder.encode([question], convert_to_numpy=True)
        q_emb = l2_normalize(q_emb).astype("float32")

        # Retrieve top-k similar chunks
        D, I = index.search(q_emb, top_k)

        retrieved_texts = []
        sources = []
        target_hash = params.get("file_hash")

        for idx in I[0]:
            if 0 <= idx < len(chunks):
                if target_hash and chunks[idx].get("file_hash") != target_hash:
                    continue

                retrieved_texts.append(chunks[idx]["text"])
                sources.append({
                    "source_file": chunks[idx]["source_file"],
                    "chunk_index": chunks[idx]["chunk_index"],
                    "text": chunks[idx]["text"]
                })

        context = "\n\n".join(retrieved_texts[:3])[:4000]
        answer = ollama_answer(question, context)
        
        if not answer or not isinstance(answer,str):
            answer = "No answer could be generated."
        
        answer = answer.strip()
        def clean_text(text):
            return text.encode("utf-8", "ignore").decode("utf-8")
        answer = clean_text(answer)
        context = clean_text(context)
        print("ANSWER RAW:", repr(answer))

        return {
            "status": "success",
            "answer": answer if answer else "No answer generated.",
            "sources": sources,
            "retrieved_context": context,
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}
