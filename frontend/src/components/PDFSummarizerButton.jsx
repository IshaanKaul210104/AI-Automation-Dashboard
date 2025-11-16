import React, { useState } from "react";
import { uploadFile } from "../api/api";
import { FileText, Upload, Loader2, Download } from "lucide-react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const PDFSummarizerButton = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [summaryPath, setSummaryPath] = useState(null);
  const [status, setStatus] = useState("");

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setSummaryPath(null);
    setStatus("");
  };

  const handleRunSummarizer = async () => {
    if (!selectedFile) {
      setStatus("⚠️ Please select a PDF file first.");
      return;
    }

    setLoading(true);
    setStatus("⏳ Uploading and processing PDF...");

    try {
      const response = await uploadFile("pdf_summarizer", selectedFile);

      if (response.status === "success") {
        setStatus("✅ PDF summarized successfully!");
        setSummaryPath(`/${response.summary_file}`);
      } else if (response.status === "duplicate") {
        setStatus(`⚠️ ${response.message}`);
      }
      else {
        setStatus(`❌ Failed: ${response.error}`);
      }
    } catch (error) {
      console.error(error);
      setStatus("❌ An error occurred while summarizing the PDF.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!summaryPath) return;
    const filename = summaryPath.split("/").pop();

    const response = await axios.get(`${API_BASE}${summaryPath}`, {
      responseType: "blob",
    });

    const blob = new Blob([response.data], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = window.URL.createObjectURL(blob);
    link.download = filename;
    link.click();
  };

  return (
    <div className="p-4 border rounded-2xl shadow-md bg-white flex flex-col items-center gap-3 w-full max-w-md">
      <div className="flex items-center gap-2 text-lg font-semibold">
        <FileText className="w-6 h-6 text-indigo-600" />
        <span>🧾 PDF Content Extractor</span>
      </div>

      <label
        htmlFor="pdf-upload"
        className="flex items-center gap-2 cursor-pointer px-4 py-2 border border-indigo-500 rounded-xl text-indigo-600 hover:bg-indigo-50 transition-all"
      >
        <Upload className="w-5 h-5" />
        <span>{selectedFile ? selectedFile.name : "Choose PDF File"}</span>
      </label>

      <input
        type="file"
        id="pdf-upload"
        accept="application/pdf"
        className="hidden"
        onChange={handleFileChange}
      />

      <button
        onClick={handleRunSummarizer}
        disabled={loading}
        className="flex items-center gap-2 px-5 py-2 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 transition-all"
      >
        {loading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            <span>Processing...</span>
          </>
        ) : (
          <>
            <FileText className="w-5 h-5" />
            <span>Run Summarizer</span>
          </>
        )}
      </button>

      {status && <p className="text-sm text-gray-700 mt-2">{status}</p>}

      {summaryPath && (
        <button
          onClick={handleDownload}
          className="flex items-center gap-2 px-4 py-2 mt-3 bg-green-600 text-white rounded-xl hover:bg-green-700 transition-all"
        >
          <Download className="w-5 h-5" />
          <span>Download Summary</span>
        </button>
      )}
    </div>
  );
};

export default PDFSummarizerButton;