// import "../Styles/Home.css"
import { Link } from "react-router-dom";
import { cn } from "../lib/utils.jsx";

const Home = () => {
  return (
    <div className="font-sans">
      {/* Hero Section */}
      <section className="text-center p-20 bg-gradient-to-r from-[#5A7DFE] to-[#A14DCE] text-white">
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          Your AI Legal Assistant
        </h1>
        <p className="text-lg md:text-xl mb-6">
          Get instant legal guidance, research assistance, and document analysis
          powered by <br /> advanced AI technology
        </p>
        <button className="px-6 py-3 bg-gradient-to-r from-indigo-800 to-blue-600 rounded-lg shadow-md transition">
          🚀 Get Started
        </button>
      </section>  

      {/* Services Section with Grid Background */}
      <section className="relative p-10 overflow-hidden" style={{ backgroundColor: "#152238" }}>
        {/* Grid background */}
        <div
          className={cn(
            "absolute inset-0",
            "[background-size:20px_20px]",
            "[background-image:linear-gradient(to_right,#2a3b55_1px,transparent_1px),linear-gradient(to_bottom,#2a3b55_1px,transparent_1px)]",
            // [background-image:linear-gradient(to_right,#2a3b55_1px,transparent_1px),linear-gradient(to_bottom,#2a3b55_1px,transparent_1px)]

            "dark:[background-image:linear-gradient(to_right,#262626_1px,transparent_1px),linear-gradient(to_bottom,#262626_1px,transparent_1px)]"
          )}
        />
        {/* Radial gradient for fade effect */}
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-gray-900 [mask-image:radial-gradient(ellipse_at_center,transparent_1%,black)]"></div>

        {/* Content */}
        <div className="relative z-20">
          <h2 className="text-3xl font-bold text-white text-center mb-8 border-purple-500 pb-2">
            Quick Access
          </h2>
          <div className="grid md:grid-cols-3 gap-8 text-center mt-8">
            {/* Card 1 */}
            <div className="bg-gray-500/30 border border-gray-700 rounded-2xl p-8 shadow-lg backdrop-blur-sm transition transform duration-300 hover:-translate-y-2 hover:bg-gradient-to-r hover:shadow-purple-500/50">
              <div className="flex justify-center mb-4">
                <div className="w-12 h-12 flex items-center justify-center bg-purple-600 rounded-full">
                  💬
                </div>
              </div>
              <h3 className="text-xl font-semibold text-white text-center mb-2">
                General Chat
              </h3>
              <p className="text-gray-300 text-center">
                Ask general legal questions and get instant AI-powered responses
              </p>
            </div>

            {/* Card 2 */}
            <div className="bg-gray-500/30 border border-gray-700 rounded-2xl p-8 shadow-lg backdrop-blur-sm transition transform duration-300 hover:-translate-y-2 hover:bg-gradient-to-r hover:shadow-purple-500/50">
              <div className="flex justify-center mb-4">
                <div className="w-12 h-12 flex items-center justify-center bg-green-600 rounded-full">
                  ⚖️
                </div>
              </div>
              <h3 className="text-xl font-semibold text-white text-center mb-2">
                Legal Advocate Agent
              </h3>
              <p className="text-gray-300 text-center">
                Specialized legal advocacy and detailed case analysis
              </p>
            </div>

            {/* Card 3 */}
            <Link to="/researchagent">
            <div className="bg-gray-500/30 border border-gray-700 rounded-2xl p-8 shadow-lg backdrop-blur-sm transition transform duration-300 hover:-translate-y-2 hover:bg-gradient-to-r hover:shadow-purple-500/50">
              <div className="flex justify-center mb-4">
                <div className="w-12 h-12 flex items-center justify-center bg-pink-600 rounded-full">
                  📄
                </div>
              </div>
              <h3 className="text-xl font-semibold text-white text-center mb-2">
                Research Paper Agent
              </h3>
              <p className="text-gray-300 text-center">
                Generate comprehensive legal research papers and analysis
              </p>
            </div>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
