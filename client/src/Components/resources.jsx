import React, { useState } from "react";
import "../Styles/resources.css";

const guides = [
  {
    icon: "📄",
    title: "Understanding Contracts",
    color: "#c084fc",
    description:
      "Learn the basics of contract law, key terms, and what to look for in agreements.",
  },
  {
    icon: "🏠",
    title: "Tenant Rights",
    color: "#34d399",
    description:
      "Know your rights as a tenant, including lease agreements and dispute resolution.",
  },
  {
    icon: "💼",
    title: "Employment Law Basics",
    color: "#f472b6",
    description:
      "Understanding workplace rights, discrimination, and wrongful termination.",
  },
  {
    icon: "👨‍👩‍👧‍👦",
    title: "Family Law Guide",
    color: "#f87171",
    description:
      "Navigate divorce, child custody, and support issues with confidence.",
  },
  {
    icon: "🚗",
    title: "Personal Injury Claims",
    color: "#facc15",
    description:
      "Steps to take after an accident and understanding your compensation rights.",
  },
  {
    icon: "📜",
    title: "Small Claims Court",
    color: "#a78bfa",
    description:
      "How to file and represent yourself in small claims court proceedings.",
  },
];

const faqs = [
  {
    question: "How accurate is the AI legal advice?",
    answer:
      "Our AI provides general legal information and guidance based on established legal principles. Always consult with a licensed attorney for personalized advice.",
  },
  {
    question: "Can I upload confidential documents?",
    answer:
      "Yes, you can upload documents securely. Your data is encrypted and protected.",
  },
  {
    question: "What file formats are supported?",
    answer:
      "We support PDF, DOCX, TXT, and many other common file formats.",
  },
  {
    question: "How do I export my research or chat history?",
    answer:
      "You can export your research and chat history as a PDF or text file.",
  },
];

const LegalResources = () => {
  const [openIndex, setOpenIndex] = useState(null);

  const toggleFAQ = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <div className="page">
      <h1 className="page-title">Legal Resources</h1>

      {/* Legal Guides Section */}
      <h2 className="section-title">📚 Legal Guides for Public</h2>
      <div className="guides-container">
        {guides.map((guide, index) => (
          <div className="guide-card" key={index}>
            <div
              className="guide-icon"
              style={{ background: guide.color + "20", color: guide.color }}
            >
              {guide.icon}
            </div>
            <h3 className="guide-title" style={{ color: guide.color }}>
              {guide.title}
            </h3>
            <p className="guide-description">{guide.description}</p>
            <a href="#" className="guide-link">
              Read Guide →
            </a>
          </div>
        ))}
      </div>

      {/* FAQ Section */}
      <h2 className="section-title">❓ Frequently Asked Questions</h2>
      <div className="faq-container">
        {faqs.map((faq, idx) => (
          <div className="faq-item" key={idx}>
            <button
              className="faq-question"
              onClick={() => toggleFAQ(idx)}
            >
              {faq.question}
              <span>{openIndex === idx ? "−" : "+"}</span>
            </button>
            {openIndex === idx && <div className="faq-answer">{faq.answer}</div>}
          </div>
        ))}
      </div>
    </div>
  );
};

export default LegalResources;
