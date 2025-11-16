import React, { useState } from "react";
import { runScript } from "../api/api";
import { MessageCircle, Send, Loader2 } from "lucide-react";

const AskQAButton = () => {
    const [question, setQuestion] = useState("");
    const [loading, setLoading] = useState(false);
    const [answer, setAnswer] = useState("");
    
    const handleAsk = async () => {
        if(!question.trim()){
            setAnswer("⚠️ Please type a question.");
            return;
        }
        
        setLoading(true);
        setAnswer("⏳ Thinking...");
        
        try {
            const res = await runScript("ask_qa", { question });

            if (res.status === "success") {
                setAnswer(res.answer);
            } else {
                setAnswer(`❌ Error: ${res.error}`);
            }
        } catch (err) {
            console.error(err);
            setAnswer("❌ Request failed.");
        }
        
        setLoading(false);
    };
    
    return (
        <div className="p-4 border rounded-2xl shadow-md bg-white flex flex-col gap-3 w-full max-w-md mt-6">
            <div className="flex items-center gap-2 text-lg font-semibold">
                <MessageCircle className="w-6 h-6 text-blue-600" />
                <span>🤖 Ask Questions About This PDF</span>
            </div>

            <textarea
                className="w-full border rounded-xl p-3 text-sm"
                placeholder="Ask a question about the uploaded PDF..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                rows={3}
            ></textarea>

            <button
                onClick={handleAsk}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-all"
            >
                {loading ? (
                <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Processing...</span>
                </>
                ) : (
                <>
                    <Send className="w-5 h-5" />
                    <span>Ask PDF</span>
                </>
                )}
            </button>

            {answer && (
                <div className="p-3 bg-gray-100 rounded-xl text-sm whitespace-pre-wrap">
                {answer}
                </div>
            )}
        </div>
    );
};

export default AskQAButton;