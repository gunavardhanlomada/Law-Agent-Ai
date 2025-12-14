import { useState } from "react";
import { NavLink } from "react-router-dom";
import "../Styles/NavBar.css";

export default function NavBar() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="navbar">
      <div className="navbar-container">
        {/* Logo */}
        <div className="navbar-logo">
          <div className="logo-icon">
            <span>⚖️</span>
          </div>
          <span className="logo-text">LegalAI Assistant</span>
        </div>

        {/* Links */}
        <div className={`navbar-links ${menuOpen ? "open" : ""}`}>
          <NavLink to="/" className="nav-link" onClick={() => setMenuOpen(false)}>
            Home
          </NavLink>
          <NavLink to="/legalsearch" className="nav-link" onClick={() => setMenuOpen(false)}>
            Legal Search
          </NavLink>
          <NavLink to="/aichat" className="nav-link" onClick={() => setMenuOpen(false)}>
            AI Chat
          </NavLink>
          <NavLink to="/researchagent" className="nav-link" onClick={() => setMenuOpen(false)}>
            Research Agent
          </NavLink>
          <NavLink to="/resources" className="nav-link" onClick={() => setMenuOpen(false)}>
            Resources
          </NavLink>
        </div>

        {/* SVG Hamburger Button (Mobile Only) */}
        <button
          className="hamburger-btn"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 50 50"
            className="hamburger-icon"
          >
            <path d="M 5 8 A 2.0002 2.0002 0 1 0 5 12 L 45 12 A 2.0002 2.0002 0 1 0 45 8 L 5 8 z M 5 23 A 2.0002 2.0002 0 1 0 5 27 L 45 27 A 2.0002 2.0002 0 1 0 45 23 L 5 23 z M 5 38 A 2.0002 2.0002 0 1 0 5 42 L 45 42 A 2.0002 2.0002 0 1 0 45 38 L 5 38 z"></path>
          </svg>
        </button>
      </div>
    </nav>
  );
}
