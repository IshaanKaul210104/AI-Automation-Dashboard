import whisper
from fastapi import UploadFile
import uuid
import os

# Load Whisper once
model = whisper.load_model("base")

def transcribe_audio(file: UploadFile):
    # Save audio temporarily
    temp_audio = f"temp_{uuid.uuid4().hex}.wav"
    with open(temp_audio, "wb") as f:
        f.write(file.file.read())

    # Transcribe
    try:
        result = model.transcribe(temp_audio)
        text = result["text"].strip()
    finally:
        os.remove(temp_audio)

    # Write transcript to a downloadable .txt file
    txt_filename = f"transcript_{uuid.uuid4().hex}.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(text)

    return {
        "status": "file",
        "filename": txt_filename
    }