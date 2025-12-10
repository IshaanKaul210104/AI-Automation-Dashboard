// src/components/ModelRecommenderButton.jsx
import React, { useState } from "react";
import axios from "axios";

export default function ModelRecommenderButton() {
  const [file, setFile] = useState(null);
  const [task, setTask] = useState("regression");
  const [targetCol, setTargetCol] = useState("");
  const [previewColumns, setPreviewColumns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setPreviewColumns([]);
    setTargetCol("");
    setResult(null);
    setStatus("");
    // quick preview: read top row to get columns
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const text = ev.target.result;
        const rows = text.split(/\r?\n/);
        const headers = rows[0].split(",");
        setPreviewColumns(headers.map(h => h.trim()));
      } catch {
        setPreviewColumns([]);
      }
    };
    reader.readAsText(f, "utf-8");
  };

  const handleRun = async () => {
    if (!file) {
      setStatus("⚠️ Choose a CSV file first.");
      return;
    }
    if ((task === "regression" || task === "classification") && !targetCol) {
      setStatus("⚠️ Select a target column for the chosen task.");
      return;
    }
    setLoading(true);
    setStatus("⏳ Uploading and running model recommender...");
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);
      // pass task and target as query params for convenience
      const res = await axios.post(`http://localhost:8000/run/model_recommender?task=${task}&target_col=${encodeURIComponent(targetCol || "")}`, form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
      });

      setResult(res.data);
      if (res.data.status !== "success") {
        setStatus(`❌ Failed: ${res.data.error || "Unknown error"}`);
      } else {
        setStatus("✅ Recommendation complete.");
      }
    } catch (err) {
      console.error(err);
      setStatus("❌ Request failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 border rounded-xl bg-white shadow-md w-full max-w-lg">
      <h3 className="text-lg font-semibold mb-2">🧠 ML Model Recommender</h3>

      <input type="file" accept=".csv" onChange={handleFileChange} className="mb-2" />

      <div className="flex gap-2 mb-2">
        <label className="flex items-center gap-1">
          <input type="radio" checked={task==="regression"} onChange={() => setTask("regression")} /> Regression
        </label>
        <label className="flex items-center gap-1">
          <input type="radio" checked={task==="classification"} onChange={() => setTask("classification")} /> Classification
        </label>
        <label className="flex items-center gap-1">
          <input type="radio" checked={task==="clustering"} onChange={() => setTask("clustering")} /> Clustering
        </label>
      </div>

      { (task === "regression" || task === "classification") && (
        <div className="mb-2">
          <label className="text-sm">Select target column</label>
          <select value={targetCol} onChange={(e)=>setTargetCol(e.target.value)} className="block w-full border rounded px-2 py-1">
            <option value="">-- choose --</option>
            {previewColumns.map((c)=> <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      )}

      <div className="flex gap-2">
        <button onClick={handleRun} disabled={loading} className="px-4 py-2 bg-indigo-600 text-white rounded">
          {loading ? "Processing..." : "Run Recommender"}
        </button>
      </div>

      {status && <p className="mt-3 text-sm text-gray-700">{status}</p>}

      {result && (
      <div className="mt-3 bg-gray-50 p-3 rounded text-sm">
        {result.status === "success" ? (
          <>
            <p><strong>Recommended model:</strong> {result.recommended_model}</p>
            <p><strong>Why:</strong> {result.reason}</p>

            <div className="mt-2">
              <strong>Meta features:</strong>
              <pre className="whitespace-pre-wrap bg-white p-2 rounded border">
                {JSON.stringify(result.meta_features, null, 2)}
              </pre>
            </div>

            <div className="mt-2">
              <strong>Metrics:</strong>
              <pre className="whitespace-pre-wrap bg-white p-2 rounded border">
                {JSON.stringify(result.metrics, null, 2)}
              </pre>
            </div>
          </>
        ) : (
          <p className="text-red-600">Error: {result.error}</p>
        )}
      </div>
    )}
    </div>
  );
}