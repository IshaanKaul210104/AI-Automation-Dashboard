import React, { useState } from "react";
import axios from "axios";

export default function AudioTranscriberButton() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  const handleRun = async () => {
    if (!file) {
      setStatus("⚠️ Please upload an audio file.");
      return;
    }

    setLoading(true);
    setStatus("⏳ Transcribing...");

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await axios.post(
        "http://localhost:8000/run/transcribe_audio",
        form,
        {
          headers: { "Content-Type": "multipart/form-data" },
          responseType: "blob", // important for downloading txt file
        }
      );

      // Create download link for .txt transcription
      const blob = new Blob([res.data], { type: "text/plain" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "transcription.txt";
      document.body.appendChild(link);
      link.click();
      link.remove();

      setStatus("✅ Transcription downloaded.");
    } catch (err) {
      console.error(err);
      setStatus("❌ Request failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 border rounded-xl bg-white shadow-md w-full max-w-lg">
      <h3 className="text-lg font-semibold mb-2">🎙️ Audio Transcriber</h3>

      <input
        type="file"
        accept="audio/*"
        onChange={(e) => setFile(e.target.files[0])}
        className="mb-2"
      />

      <button
        onClick={handleRun}
        disabled={loading}
        className="px-4 py-2 bg-purple-600 text-white rounded"
      >
        {loading ? "Processing..." : "Transcribe Audio"}
      </button>

      {status && <p className="mt-3 text-sm text-gray-700">{status}</p>}
    </div>
  );
}