import React from 'react';
// import './App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import NavBar from './Components/NavBar';
import Aichat from './Components/Aichat';
import LegalResources from './Components/resources';
import Home from './Components/Home';
import ResearchPaperAgent from './Components/ResearchPaperAgent';
import LegalSearch from './Components/LegalSearch';

function App() {

  return (
    <>
    <BrowserRouter>
    <NavBar/>
    <Routes>
      <Route path="/" element={<Home/>} />
      <Route path="/legalsearch" element={<LegalSearch/>} />
      <Route path="/aichat" element={<Aichat/>} />
      <Route path="/researchagent" element={<ResearchPaperAgent/>} />
      <Route path="/resources" element={<LegalResources />} />
      <Route path="*" element={<div>404 Not Found</div>} />
    </Routes>
    </BrowserRouter>
    </>
  )
}

export default App
