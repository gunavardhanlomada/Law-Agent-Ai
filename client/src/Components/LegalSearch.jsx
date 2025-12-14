import React, { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import "../Styles/LegalSearch.css";

const API_BASE = "http://localhost"; // 🔹 change this to your backend URL

// Suggested topics list
const topics = [
  { title: "Contract Law", desc: "Agreements, breaches, and remedies", icon: "📜" },
  { title: "Employment Rights", desc: "Workplace laws and protections", icon: "💼" },
  { title: "Property Law", desc: "Real estate and ownership rights", icon: "🏠" },
  { title: "Family Law", desc: "Divorce, custody, and support", icon: "👨‍👩‍👧" },
  { title: "Criminal Law", desc: "Offenses and legal procedures", icon: "⚖️" },
  { title: "Business Law", desc: "Corporate regulations and compliance", icon: "🏢" },
];

// API call function
const searchLaw = async (query) => {
  const res = await axios.post(`${API_BASE}/search`, { query });
  return res.data;
};

const LegalSearch = () => {
  const [search, setSearch] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (query) => {
    if (!query.trim()) return;
    setSearch(query);
    setLoading(true);
    try {
      const data = await searchLaw(query);
      setResults(data);
    } catch (error) {
      console.error("Error fetching search results:", error);
    }
    setLoading(false);
  };

  const filteredTopics = topics.filter((topic) =>
    topic.title.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="legal-search">
      <h2>Legal Topic Search</h2>
      <div className="search-bar">
        <input
          type="text"
          placeholder="Search legal topics, cases, or laws..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <button onClick={() => handleSearch(search)}>🔍 Search</button>
      </div>

      {/* Loader */}
      {loading && <p style={{ marginTop: "20px" }}>Searching...</p>}

      {/* Show results */}
      {results ? (
        <div className="topics-container">
          <h3>Search Results</h3>
          <div className="topic-card" style={{ textAlign: "left" }}>
            {/* Render Markdown result */}
            <ReactMarkdown>{results.result}</ReactMarkdown>

            {/* Official Links (optional) */}
            {results.official_links && (
              <div style={{ marginTop: "20px" }}>
                <h4 style={{ color: "#1abc9c" }}>Official Links</h4>
                <p style={{ color: "#bbb" }}>{results.official_links}</p>
              </div>
            )}
          </div>

          {/* Back button */}
          <button
            style={{
              marginTop: "20px",
              padding: "10px 20px",
              borderRadius: "8px",
              border: "none",
              background: "linear-gradient(90deg, #9b59b6, #3498db)",
              color: "white",
              cursor: "pointer",
            }}
            onClick={() => setResults(null)}
          >
            ⬅ Back to Suggested Topics
          </button>
        </div>
      ) : (
        <div className="topics-container">
          <h3>Suggested Topics</h3>
          <div className="topics-grid">
            {filteredTopics.map((topic, index) => (
              <div
                key={index}
                className="topic-card"
                onClick={() => handleSearch(topic.title)}
                style={{ cursor: "pointer" }}
              >
                <h4>
                  {topic.icon} {topic.title}
                </h4>
                <p>{topic.desc}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default LegalSearch;
