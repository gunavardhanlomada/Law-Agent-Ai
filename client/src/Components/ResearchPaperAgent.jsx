import { useState } from "react";
import axios from "axios";
import "../Styles/ResearchPaperAgent.css";

const API_BASE = "http://localhost"; 

const startPapersSession = async() => {
  const res = await axios.post(`${API_BASE}/papers/start`);
  return res.data;
};

const uploadPapers = async(sessionId, files) => {
  const formData = new FormData();
  formData.append("session_id", sessionId);
  files.forEach((file) => {
    formData.append("files", file);
  });

  const res = await axios.post(`${API_BASE}/papers/upload`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

const sendPapersMessage = async(sessionId, query) => {
  const res =  await axios.post(`${API_BASE}/papers/message`, {
    session_id: sessionId,
    query,
  });
  return res.data;
};

export default function ResearchPaperAgent() {
  const [topic, setTopic] = useState("");
  const [files, setFiles] = useState([]);
  const [output, setOutput] = useState("");
  const [sessionId, setSessionId] = useState(null);

  // handle file selection
  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  // handle research generation
  const handleGenerate = async () => {
    try {
      let session = sessionId;
      if (!session) {
        const sessionData = await startPapersSession();
        session = sessionData.session_id;
        setSessionId(session);
      }

      if (files.length > 0) {
        await uploadPapers(session, files);
      }

      const response = await sendPapersMessage(session, `Research Topic: ${topic}`);
      console.log("Research Paper Response:", response);
      setOutput(response.answer || "No output returned from server.");
    } catch (error) {
      console.error("Error generating paper:", error);
      setOutput("❌ Failed to generate research paper.");
    }
  };

  return (
    <div className="page">
      <h1 className="title">Research Paper Agent</h1>
      <div className="container">
        
        {/* Left Panel */}
        <div className="card fly-in-left">
          <h2 className="card-heading">🔬 Research Parameters</h2>

          <div className="form-group">
            <label>Research Topic</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Enter your research topic..."
            />
          </div>

          {/* File Upload */}
          <div className="form-group">
            <label className="upload-title">📎 Upload Documents</label>
            <div className="upload-box">
              <input
                type="file"
                id="fileUpload"
                multiple
                onChange={handleFileChange}
                accept=".pdf,.doc,.docx"
              />
              <label htmlFor="fileUpload" className="upload-label">
                📎 Upload PDF/DOCX
              </label>
            </div>
            {files.length > 0 && (
              <ul className="file-list">
                {files.map((file, idx) => (
                  <li key={idx}>{file.name}</li>
                ))}
              </ul>
            )}
          </div>

          <button className="btn-generate" onClick={handleGenerate}>
            🔍 Generate Research Paper
          </button>
        </div>

        {/* Right Panel */}
        <div className="card fly-in-right">
          <div className="card-header">
            <h2 className="card-heading">📑 Research Output</h2>
            <button className="btn-export">Export PDF</button>
          </div>
          <div className="output-box"  dangerouslySetInnerHTML={{__html:output}}>
            {/* {"Your research paper will appear here after generation..."} */}
          </div>
        </div>
      </div>
    </div>
  );
}
