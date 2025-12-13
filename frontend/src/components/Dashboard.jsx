import React from "react";
import WebScraperButton from "./WebScraperButton";
import PDFSummarizerButton from "./PDFSummarizerButton";
import AskQAButton from "./AskQAButton";
import MLModelRecommender from "./MLModelRecommender";
import AudioTranscriberButton from "./AudioTranscriberButton";

export default function Dashboard() {
  return (
    <div className="p-6 text-center">
      <h1 className="text-3xl font-bold mb-8">Automation Dashboard</h1>

      {/* 3-column layout */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">

        {/* LEFT COLUMN – Web Scraper + Audio Transcriber */}
        <div className="flex flex-col items-center space-y-6">
          <WebScraperButton />

          {/* 👇 NEW BUTTON EXACTLY BELOW WEB SCRAPER */}
          <AudioTranscriberButton />
        </div>

        {/* CENTER COLUMN – PDF summarizer + QA */}
        <div className="flex flex-col items-center space-y-6">
          <PDFSummarizerButton />
          <AskQAButton />
        </div>

        {/* RIGHT COLUMN – Model Recommender */}
        <div className="flex flex-col items-center">
          <MLModelRecommender />
        </div>

      </div>
    </div>
  );
}