import React, { useState } from "react";
import { runScript } from "../api/api";
import { API_BASE } from "../api/api";

// 🔹 Preset websites
const PRESET_SITES = {
  "": "Custom (enter your own URL)",
  "https://www.theverge.com": "The Verge",
  "https://medium.com": "Medium",
  "https://towardsdatascience.com": "Towards Data Science",
  "https://www.cnn.com": "CNN",
  "https://en.wikipedia.org/wiki/Main_Page": "Wikipedia",
  "https://techcrunch.com": "TechCrunch",
  "https://www.bbc.com/news": "BBC News",
  "https://news.ycombinator.com": "Ycombinator News"
};

export default function WebScraperButton() {
  const [status, setStatus] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [downloadUrl, setDownloadUrl] = useState("");
  const [selectedSite, setSelectedSite] = useState(""); // 🔹 NEW

  const handleClick = async () => {
    setStatus("⏳ Running Web Scraper...");
    setDownloadUrl("");

    // If URL box is empty, fallback to selected site or default
    const url =
      inputValue.trim() ||
      selectedSite ||
      "https://www.theverge.com";

    try {
      const res = await runScript("webscraper", { url });

      if (res?.status === "failed") {
        setStatus("❌ Web Scraper failed — check backend logs.");
        return;
      }

      const count = res?.output?.count ?? res?.count ?? 0;
      setStatus(`✅ Web Scraper completed — Articles: ${count}`);

      const outputDir = res?.output?.output ?? res?.output ?? "";
      if (outputDir) {
        const csvUrl = `${API_BASE}/${outputDir.replace(/\\/g, "/")}/articles.csv`;
        setDownloadUrl(csvUrl);
      }
    } catch (err) {
      console.error(err);
      setStatus("❌ Web Scraper failed — check backend logs.");
    }
  };

  return (
    <div className="flex flex-col items-center bg-white p-4 rounded-lg shadow-md w-96">
      <h2 className="text-lg font-semibold mb-3">📰 Article Web Scraper</h2>

      {/* 🔹 NEW: Dropdown */}
      <select
        value={selectedSite}
        onChange={(e) => {
          setSelectedSite(e.target.value);
          setInputValue(e.target.value); // auto-fill input box
        }}
        className="border border-gray-300 rounded-md px-3 py-2 mb-2 w-full text-sm"
      >
        {Object.entries(PRESET_SITES).map(([url, label]) => (
          <option key={url} value={url}>
            {label}
          </option>
        ))}
      </select>

      {/* Existing URL input */}
      <input
        type="text"
        placeholder="Enter website URL (or choose from dropdown)"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        className="border border-gray-300 rounded-md px-3 py-2 mb-2 w-full text-sm"
      />

      <button
        onClick={handleClick}
        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
      >
        Run Web Scraper
      </button>

      {status && (
        <p className="mt-3 text-sm text-gray-700 text-center">{status}</p>
      )}

      {downloadUrl && (
        <a
          href={downloadUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-3 bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition"
        >
          ⬇️ Download CSV
        </a>
      )}
    </div>
  );
}