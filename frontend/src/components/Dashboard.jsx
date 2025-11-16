import React from "react";
import WebScraperButton from "./WebScraperButton";
import PDFSummarizerButton from "./PDFSummarizerButton";
import AskQAButton from "./AskQAButton";
import RunButton from "./RunButton";

export default function Dashboard() {
  return (
    <div className="p-6 text-center">
      <h1 className="text-3xl font-bold mb-8">Automation Dashboard</h1>

      {/* 3-column layout */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">

        {/* LEFT COLUMN – Web Scraper */}
        <div className="flex flex-col items-center">
          <WebScraperButton />
        </div>

        {/* CENTER COLUMN – PDF summarizer + QA stacked */}
        <div className="flex flex-col items-center space-y-6">
          <PDFSummarizerButton />
          <AskQAButton />
        </div>

        {/* RIGHT COLUMN – Final functionality */}
        <div className="flex flex-col items-center">
          <RunButton scriptName="github_fetcher" label="Fetch GitHub Stats" />
        </div>
        
      </div>
    </div>
  );
}