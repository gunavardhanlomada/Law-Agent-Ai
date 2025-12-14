import os
import streamlit as st
import google.generativeai as genai
from phi.agent import Agent
from phi.model.google import Gemini
from pydantic import BaseModel, Field
from phi.tools.googlesearch import GoogleSearch
from phi.tools.website import WebsiteTools 
from phi.tools.wikipedia import WikipediaTools 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_option_menu import option_menu
import re
import json
import uuid

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
sa = os.getenv("SERP_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

if 'user_index_name' not in st.session_state:
    st.session_state.user_index_name = f"legal-qa-db-{uuid.uuid4().hex[:8]}"
    
def del_index(index_name):
    pc.delete_index(name=index_name)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "legal-qa-db"

if st.session_state.user_index_name not in [i['name'] for i in pc.list_indexes()]:
    pc.create_index(
        name=st.session_state.user_index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'),
    )

if "index" not in st.session_state:
    st.session_state.index = pc.Index(st.session_state.user_index_name)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    vector_store = PineconeVectorStore(index = st.session_state.index, embedding=embeddings)
    vector_store.add_texts(text_chunks)
    return vector_store

def get_rel_text(user_question, db):
    docs = db.similarity_search(user_question, k = 1)
    return docs[0].page_content

research_agent = Agent(
    name="Legal Research PA",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[GoogleSearch(), WebsiteTools(), WikipediaTools()],
    description="You are a virtual research assistant specializing in Indian law.",
    instructions=[
        "Identify related statutes, IPC sections, landmark cases.",
        "Summarize cases with citations and official links.",
        "Structure detailed research reports with headings and resources.",
        "Do not provide hypothetical examples or case studies.",
        "Search for the latest and most relevant information.",
        "Provide results in markdown format with headings and subheadings.",
        "Do not include any disclaimers or unnecessary information.",

         "NOTE : Do not use indiankanoon website for the search."
         "Provide the SCC citations for all the cases.",
         "Search through the web for citation of cases and all"
         
    ],
    markdown=True,
    show_tool_calls=False
)

website_agent = Agent(
    name="Legal Website Agent",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[WebsiteTools(),GoogleSearch()],
    description="Fetch official government/legal website links and PDF documents.",
    instructions=[
        "Provide official links for cases and citations.",
        "Search for PDF documents from the web",
        "Search for the official links of the cases using citations or the necessary points given from the Indian government website ''Casemine', 'sci.gov.in','Drishti Judiciary', 'Delhi Judicial Academy', etc' where the user can find the cases and citations."
        "Provide the citations and links for the cases and statutes.", 
        "Try to provide the links of cases from Casemine, and other official websites like 'Drishti Judiciary, Delhi Judicial Academy, etc'",
        "Search for the latest and most relevant information through the web and using your knowledge.",
        
        "Provide the available SCC citations for all the cases.",

        "NOTE : Do not use indiankanoon website for the search.",   
        "Do not provide hallucinations or irrelevant information.",      
        "Search through the web for citation of cases and all"  

        "Do not provide like search strategies and all, you only perform the search and return the results."  
   
    ],
    markdown=True,
    show_tool_calls=False
)

lawer_agent = Agent(
    name="Lawyer PA",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[GoogleSearch(), WebsiteTools(), WikipediaTools()],
    description=(
        "You are a professional legal AI agent specialized in Indian law, "
        "trained on Phi data and capable of researching, analyzing, and arguing "
        "hypothetical cases. Your goal is to leverage precedents, statutes, IPC sections, "
        "and landmark judgments to construct well-reasoned arguments and research reports "
        "for any user-provided scenario."
    ),
    instructions=[        
        "When a user presents a hypothetical or real case fact pattern, identify the key legal issues and relevant facts.",        
        "Search for analogous Indian court decisions, statutes, and IPC sections using provided tools (GoogleSearch, WebsiteTools, WikipediaTools).",        
        "For each relevant precedent, extract: parties, facts, legal issues, reasoning, outcome, and citations (with official links). Summarize concisely with headings.",
        "Organize the research report into clear sections: (a) Facts & Issues; (b) Relevant Statutes/IPC Sections; (c) Precedent Summaries; (d) Argument Outline; (e) Possible Counterarguments; (f) Conclusion & Recommendations.",
        "Based on analogous cases, construct logical arguments for the hypothetical scenario—identifying strengths, weaknesses, and potential defenses—while citing statutes and case law.",
        "Incorporate insights from Phi-trained legal knowledge to ensure depth, accuracy, and contextual understanding in your analysis.",
        "Always include full citations (case names, report numbers, dates) and direct URLs to primary sources (e.g., Supreme Court or High Court databases) in a ‘References’ section.",
        "Deliver the final output in Markdown format with nested headings, bullet points, and numbered lists for readability.",
        "Prioritize the most recent and relevant information; verify dates and publication sources to ensure currency.",
        "Avoid adding disclaimers, personal opinions, or unrelated commentary. Focus strictly on legal analysis."

        "NOTE : Do not use indiankanoon website for the search."
        "Provide the SCC citations for all the cases."
        "Search through the web for citation of cases and all"
    ],
    markdown=True,
    show_tool_calls=False
)

research_paper_agent = Agent(
    name="Legal Research Paper Agent",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[GoogleSearch(), WebsiteTools(), WikipediaTools()],
    description = (
    "You are an AI agent specialized in gathering, summarizing, and synthesizing "
    "legal research papers and journal articles. When provided with a base paper "
    "(PDF or text) and a user query, you will:\n"
    "  1. Extract and index the base paper’s text into thematic chunks.\n"
    "  2. Search legal journals and databases for all existing research papers "
    "     that relate both to the base paper’s topic and the user’s query.\n"
    "  3. For each related paper, provide:\n"
    "       • Full citations in Bluebook AND OSCOLA formats.\n"
    "       • A stable link or DOI/URL.\n"
    "       • A brief summary showing how it aligns with the base paper and user query.\n"
    "       • Excerpts of the portions (sections, paragraphs) that are directly relevant to the base paper and user query.\n"
    "  4. On demand, generate a mapping or explanation of how each scraped paper connects conceptually to the base paper and user query.\n"
    "\n"
    "All responses must be delivered in Markdown with nested headings, lists, and clear citation placeholders."
),

instructions = [
    "When a user uploads one or more PDF papers, automatically extract their text and split it into thematic chunks (e.g., Abstract, Introduction, Literature Review, Analysis, Conclusion). "    

    "Provide a minimum of 25 papers related to the base paper, and also the papers related to the query given by the user, and arrange them in the Top-k order of relevance based on the user’s query.",
    "Upon a user query (e.g., 'Find ALL papers related to digital evidence in India based on this base paper'), retrieve the top-k relevant chunks from the indexed base paper. "
    "Use these chunks plus the user’s instructions to formulate search queries. Then use the GoogleSearch() tool (or other approved academic search tools) to locate all legal journal articles, SSRN papers, and conference proceedings that: "
    "  a. Share themes, methodologies, or citations with the base paper. "
    "  b. Address the specific angle of the user’s query. "
    "  c. Are published in reputable journals or conferences. "
    "For each paper found, collect: Title, Authors, Year, Venue, DOI/URL, and, if applicable, case citation (SCC).",
    "Provide all research papers related to the base paper given by the user, and also the papers related to the query given by the user.",
    
    "For every related paper (articles and cases), generate:\n"
    "  • A Bluebook-style citation (e.g., 'John Doe, Digital Evidence and Indian Law, 12 J. Indian L. 45 (2018)').\n"
    "  • An OSCOLA-style citation (e.g., 'John Doe, ‘Digital Evidence and Indian Law’ (2018) 12 Journal of Indian Law 45').\n"
    "Include both formats in bullet points under each paper heading.",

    "Under each related paper’s heading, provide:\n"
"  1. A one-paragraph summary explaining how the paper aligns with the base paper’s themes and/or the user’s query.\n"
"  2. Precise excerpted text (section or paragraph) from that paper which directly corresponds to the base paper’s arguments or the user’s question. "
"     Place excerpts inside blockquotes to distinguish them from your own explanatory text.\n"
"  3. The stable link or DOI/URL from the legal journal (or conference proceeding) to access the full paper directly.\n"
"  4. If no direct PDF/DOI/URL is available online, then provide the public URL of the journal’s homepage or the specific issue page where that article is listed (not a hypothetical link). \n"
"     Do NOT hypothetically or invent or guess any URLs—only use real, verifiable links.\n\n"

    "Deliver your answer in Structured format with nested headings. For example:\n"
    "  ## Related Papers for [User Query]\n"
    "  ### 1. Title of Paper A\n"
    "   - **Bluebook:** …\n"
    "   - **OSCOLA:** …\n"
    "   - **Link/DOI:** …\n"
    "   - **Summary:** …\n"
    "   - **Relevant Excerpt:**\n"
    "    > …\n"
    "\n"
    "  ### 2. Title of Paper B\n"
    "   - **Bluebook:** …\n"
    "   - **OSCOLA:** …\n"
    "   - **Link/DOI:** …\n"
    "   - **Summary:** …\n"
    "   - **Relevant Excerpt:**\n"
    "    > …",

    "If the user explicitly asks for a detailed mapping between the base paper and the scraped papers, produce a separate section:\n"
    "  ### Relationship Map\n"
    "  - **Base Paper Section:** [e.g., Literature Review]\n"
    "    - **Related Paper A:** Explanation of shared concepts or citations.\n"
    "    - **Related Paper B:** Explanation of methodological overlap or contrasting findings.\n"
    "  - **Base Paper Section:** [e.g., Methodology]\n"
    "    - **Related Paper A:** …\n"
    "    - **Related Paper C:** …",

    "When searching external sources, avoid indiankanoon. Instead, prioritize Google Scholar, SSRN, HeinOnline, Westlaw India (if accessible), JSTOR, and university repositories. "
    "Verify publication dates and venues before including any paper. Prioritize literature published within the last 10 years (2015–2025) unless a seminal older work is explicitly relevant.",

    "Whenever referencing case law, always include the official SCC citation in Bluebook and OSCOLA formats. "
    "For example:\n"
    "  - **Bluebook (Case):** State of Maharashtra v. Mayer Hans George, (2008) 2 SCC 1.\n"
    "  - **OSCOLA (Case):** State of Maharashtra v Mayer Hans George [(2008) 2 SCC 1].",

    "Maintain a neutral, scholarly tone. Do not include personal opinions, unrelated commentary, or marketing language. Focus strictly on legal research and academic writing guidance.",

    "All outputs must be in Structured format with nested headings, bullet lists, and clearly labeled citation sections. "
    "Do not include personal disclaimers or extraneous text. If the user requests full-text excerpts beyond the relevant passages, ask for confirmation to ensure compliance with copyright."
],

    markdown=True,
    show_tool_calls=False
)

class RelevanceScoreTool(BaseModel):
    score : int = Field(..., description="The Score of the relevance between the base paper and the related paper with respect to the user query, in percentage (0-100%)")

relevance_score_agent = Agent(
    name="Relevance Score Agent",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[],
    description=(
        "You are an AI agent specialized in computing a relevance score between a base legal research paper "
        "and a single related paper, with respect to a specific user query. Each call evaluates one related paper "
        "against the base paper content and the query, and returns a numeric relevance score (0–100%) plus a brief rationale."
    ),
    instructions=[
        
        "Each prompt will include:\n"
        "  • Base Paper Content (relevant thematic chunks or summary).\n"
        "  • Related Paper Details:\n"
        "       Title\n"
        "       Bluebook Citation\n"
        "       OSCOLA Citation\n"
        "       Link/DOI\n"
        "       Summary (1 paragraph)\n"
        "       Relevant Excerpt (blockquoted text)\n"
        "  • User Query",

        "Use the user query as context. Compare the related paper’s summary and excerpt against the base paper content. "
        "Evaluate:\n"
        "  • Keyword overlap between base paper and related paper excerpt, focused on query terms.\n"
        "  • Thematic similarity: shared legal concepts, statutes, case citations, or methodology.\n"
        "  • Direct citation or explicit reference to the base paper’s core arguments (if present).\n"
        "Combine these factors to generate a relevance score from 0 to 100%.",

        "Return your answer in Markdown using this structure:\n"
        "```\n"
        "### Title: [Related Paper Title]\n"
        "- **Bluebook:** [Bluebook Citation]\n"
        "- **OSCOLA:** [OSCOLA Citation]\n"
        "- **Link/DOI:** [URL or DOI]\n"
        "- **Summary:** [One-paragraph summary]\n"
        "- **Relevant Excerpt:**\n"
        "  > [Excerpt text]\n"
        "- **Relevancy Score:** [XX%]\n"
        "  **Rationale:** [2–3 sentences explaining why this score was assigned]\n"
        "```\n"
        "Do not include any other headings or sections.",

        "When assigning numeric values, consider:\n"
        "  • ≥ 80%: Very high alignment—majority of base paper concepts and query aspects are reflected.\n"
        "  • 60–79%: Moderate alignment—significant overlap in themes or methods.\n"
        "  • 40–59%: Low–moderate alignment—some shared keywords or citations but limited depth.\n"
        "  • < 40%: Minimal alignment—superficial or tangential connection only.",

        "Maintain a concise, scholarly tone. Focus strictly on relevance between the base paper and this related paper with respect to the user query. Do not include personal opinions or unrelated commentary."
    ],
    markdown=True,
    show_tool_calls=False,
    response_model=RelevanceScoreTool
)

def fetch_official_links(query: str) -> str:
    try:
        prompt = f"Given the dictionary {query} consisting of the cases or the necessary names and citations, Search for the official links of the cases using citations or the necessary points given from the Indian government website 'Casemine','sci.gov.in', and other web pages, Also provide the official and available SCC citations for the cases and statutes. " 
        result = website_agent.run(prompt)
        return result.get_content_as_string()
    except Exception as e:
        return f"_Error fetching official links: {e}_"
    
def parse_gemini_response(response_text):
    try:
        cleaned = re.sub(r'```json|```', '', response_text)
        json_str = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_str:
            return json.loads(json_str.group())
        return None

    except Exception as e:
        st.error(f"Failed to parse response: {str(e)}")
        return None
    
def bot_response(model, query, relevant_texts, history):     

    context = ' '.join(relevant_texts)
    prompt = f"""This is the context of the document 
    Context: {context}
    And this is the user query
    User: {query}
    And this is the history of the conversation
    History: {history}

    YOU ARE A LEGAL RESEARCH ASSISTANT, AND YOU ARE A PROFESSIONAL PERSON IN THE LEGAL DOMAIN, AND YOU ARE RESPONSIBLE FOR PROVIDING ACCURATE AND RELIABLE ANSWERS TO THE USER (RELATED TO INDIAN LAW)

    Please generate a response to the user query based on the context and history
    The questions might be asked related to the provided context, and may also be in terms of the external content related to the document,
    Answer the query with respect to the context provided, you can also use your additional knowledge too, but do not ignore the content of the provided context,
    Answer the queries like a professional person being in the domain of LEGAL Authority of INDIAN LAW, having a lot of knowledge on the LAW and LEGISLATION of the INDIAN GOVERNMENT

    Please try to provide the lists of the cases and citations from the official government website ''Casemine', 'sci.gov.in','Drishti Judiciary', 'Delhi Judicial Academy', etc' where the user can find the cases and citations.",'

    "For the respoonse Do not include any disclaimers or unnecessary information."
    "Do not provide like search strategies and all, you only perform the search and return the results."
    "Do not provide like As per your request, I have searched the web and all, you only perform the search and return the results."
    "Do not provide like Okay assistant, I have searched the web and all, you only perform the search and return the results."
    
    "NOTE : Do not use indiankanoon website for the search."
    "For premium websites, or the websites which are not free, do not provide the links for citations, or provide the citations, like SCC citation, manupatra citation and all",      

    Bot:
    """

    agent_sol = research_agent.run(prompt).get_content_as_string()    

    response = model.generate_content(
        f"Prompt : {prompt}\n Answer the given prompt, additionally you can use the knowledge from agent which is {agent_sol}",
        generation_config=genai.GenerationConfig(
            temperature=0.2,
        )
    ).text

    if response:
        contr = genai.GenerativeModel("gemini-2.0-flash").generate_content(f"""
                    Provided the information about the case and statute, law or any information regarding the legal domain extract the cases and their citations, or any important keywords and statement from the text.
                    Give it in a STRICT JSON 
                    format, with the following

                    "Case Name" : [List of string of cases],
                    "Citation" : [List of string of citations],   
                    "Necessary Content" : [List of string of necessary content],                                                                                                                                                                                                                                                                      

                    The content is {response}  
                        
                    """).text
        
        dict_ = parse_gemini_response(contr)
        links = fetch_official_links(f"The provided is the dictionary of cases and citations {dict_}")
        response = f"{response}\n\n---\n**Official Sources:**\n{links}"

    return response    


def agent_bot_response(model, query, relevant_texts, history):     

    context = ' '.join(relevant_texts)
    prompt = f"""This is the context of the document:
        Context: {context}

        This is the user query:
        User: {query}

        This is the history of the conversation:
        History: {history}

        YOU ARE A LEGAL RESEARCH ASSISTANT SPECIALIZING IN INDIAN LAW. YOU ARE RESPONSIBLE FOR PROVIDING ACCURATE, RELIABLE, AND CONTEXTUAL ANSWERS TO THE USER, DRAWING FROM THE PROVIDED DOCUMENT CONTEXT, THE CONVERSATION HISTORY, AND YOUR EXPERTISE IN INDIAN LEGAL AUTHORITIES.

        INSTRUCTIONS:
        1. Analyze the query in light of the given context and history. Prioritize information from “Context” but supplement with your broader knowledge of Indian statutes, IPC sections, and landmark judgments.
        2. Identify and list all relevant cases, statutory provisions, and IPC sections. Provide full citations (case name, citation number, court, year) and direct URLs to primary sources (e.g., Indian government website ''Casemine', 'sci.gov.in','Drishti Judiciary', 'Delhi Judicial Academy', etc' where the user can find the cases and citations.",'") whenever available.
        3. Structure your response clearly with Markdown headings and subheadings:
        - Facts & Issues
        - Relevant Statutes/Sections
        - Precedent Summaries
        - Legal Analysis & Argument
        - Conclusion
        - References
        4. Summaries of precedents must include: parties, facts, legal issues, reasoning, outcome, and citation.
        5. Do not include any disclaimers, personal opinions, or unrelated commentary. Focus strictly on legal analysis and citations.
        6. Do not describe your search strategy or mention that you used external tools. Only present the findings.
        7. When citing ''Casemine', 'sci.gov.in','Drishti Judiciary', 'Delhi Judicial Academy', etc' where the user can find the cases and citations.", ensure the citation links are accurate and point directly to the relevant judgment or article.
        8. Use professional legal language appropriate for Indian law practitioners. Maintain a neutral, authoritative tone.

        "NOTE : Do not use indiankanoon website for the search."
        "For premium websites, or the websites which are not free, do not provide the links for citations, or provide the citations, like SCC citation, manupatra citation and all",      

        Bot:
        """

    agent_sol = lawer_agent.run(prompt).get_content_as_string()    

    response = model.generate_content(
        f"Prompt : {prompt}\n Answer the given prompt, additionally you can use the knowledge from agent which is {agent_sol}",
        generation_config=genai.GenerationConfig(
            temperature=0.2,
        )
    ).text

    if response:
        contr = genai.GenerativeModel("gemini-2.0-flash").generate_content(f"""
                    Provided the information about the case and statute, law or any information regarding the legal domain extract the cases and their citations, or any important keywords and statement from the text.
                    Give it in a STRICT JSON 
                    format, with the following

                    "Case Name" : [List of string of cases],
                    "Citation" : [List of string of citations],   
                    "Necessary Content" : [List of string of necessary content],                                                                                                                                                                                                                                                                      

                    The content is {response}  
                            
                    """).text
        
        dict_ = parse_gemini_response(contr)
        links = fetch_official_links(f"The provided is the dictionary of cases and citations {dict_}")
        response = f"{response}\n\n---\n**Official Sources:**\n{links}"

    return response   

def research_paper_agent_response(agent, query, relevant_text, history):
    context_block = relevant_text if relevant_text else ""
    prompt = f"""Context from Base Paper (RAG retrieved):
{context_block}

User Query:
{query}

Conversation History:
{history}

Relevance Score Agent: 

"Use the user query as context. Compare the related paper’s summary and excerpt against the base paper content. "
        "Evaluate:\n"
        "   Keyword overlap between base paper and related paper excerpt, focused on query terms.\n"
        "   Thematic similarity: shared legal concepts, statutes, case citations, or methodology.\n"
        "   Direct citation or explicit reference to the base paper’s core arguments (if present).\n"
        "Combine these factors to generate a relevance score from 0 to 100%.",

"When assigning numeric values, consider:\n"
        "   ≥ 80%: Very high alignment—majority of base paper concepts and query aspects are reflected.\n"
        "   60–79%: Moderate alignment—significant overlap in themes or methods.\n"
        "   40–59%: Low–moderate alignment—some shared keywords or citations but limited depth.\n"
        "   < 40%: Minimal alignment—superficial or tangential connection only.",

------------        

YOU ARE A LEGAL RESEARCH ASSISTANT AGENT. FOLLOW THESE STEPS STRICTLY:
1. Use the provided base-paper context above (if any) to ground your answer.
2. "Provide atleast 25 papers related to the base paper, and also the papers related to the query given by the user, and arrange them in the Top-k order of relevance based on the user’s query.",
2. If the query explicitly asks for related literature beyond the base paper, initiate a literature search using GoogleSearch() (or equivalent academic search tools) to find relevant journal articles, SSRN papers, or conference proceedings that:
   a. Share themes, methodologies, or citations with the base paper.
   b. Address the specific angle of the user’s query.
   For each found item, extract:
     - Title
     - Authors
     - Publication Venue & Year
     - DOI or Stable URL
     - Bluebook‐style citation
     - OSCOLA‐style citation
     - One‐paragraph summary focusing on how it relates to the base paper and user query
     - Relevant excerpt(s) (section or paragraph) from the paper that map to the base paper’s arguments or the query, enclosed in blockquotes.
3. Structure your response in Markdown with headings/subheadings:
   ## Summary of Base Paper Sections
   (e.g., Abstract, Key Findings)
   ## External Literature Findings (if applicable)
     ### 1. Title of Paper A
      **Bluebook:** …
      **OSCOLA:** …
      **Link/DOI:** …
      **Summary:** …
      **Relevant Excerpt:**  
      **Relevancy Score:** 
       > …
     ### 2. Title of Paper B
      **Bluebook:** …
      **OSCOLA:** …
      **Link/DOI:** …
      **Summary:** …
      **Relevant Excerpt:**  
      **Relevancy Score:** 
       > …
   
4. For citations, prioritize open‐access or freely available links. If a source is paywalled, include the DOI and metadata so the user can locate it.
5. Avoid disclaimers or unrelated commentary. Focus on academic/legal research style.
6. Provide all the records in the sorted manner based on the relevance to the user query and the base paper.
7. Also proivde the papers in a sorted manner based on the priority of journals (You can use "https://www.scopus.com/sources.uri?zone=TopNavBar&origin=searchbasic" for knowing the priority of journals)

Begin your response below:
"""
    
    initial_output = agent.run(prompt).get_content_as_string()            

    md_response = "### Research Agent Findings\n\n"
    md_response += initial_output + "\n\n"

    return md_response

def get_score(query):
    score = relevance_score_agent.run(query).get_content_as_string()
    return score

st.set_page_config(page_title="Legal Research Agent", layout="wide", page_icon="⚖️")
st.title("Indian Legal Research Agent")


with st.sidebar:
    tab = option_menu("Select Module", ["Search Hub", "Text Chat Agent", "Legal Advocate Agent", "Research Paper Agent"], orientation="vertical", icons=["search", "chat", "person", "file-earmark-text"])

if 'vector_stores' not in st.session_state:
    st.session_state.vector_stores = {}

if tab == "Search Hub":
    st.header("🔍 Search Hub")
    query = st.text_input("Enter case/topic:")
    if st.button("Search Statutes & Cases"):
        with st.spinner("Retrieving statutes and cases..."):                
            contents=(
                    f"For the given query '{query}'",
                    f"Provide relevant Indian statutes, citation and details, IPC sections, landmark cases" 
            
                    "If the query is not related to Indian law, please say 'Not related to Indian law'."
                    "If the query is of a particular case or something, then provide that particular case's name and citation also additional to the landmark cases"                       
                    "Provide the results in markdown format with headings and subheadings."
                    "Provide the results in a structured format with headings and subheadings."
                                            
                    "Do not provide hypothetical examples or case studies."
                    "Search for the latest and most relevant information."

                    "Please go through this website for the cases and statutes: https://www.aironline.in/index.html"
                    "You can also search the cases and case information from the web and using your knowledge also"

                    "You should search the cases and case information from this website I provided as well as others and then take the results from there."
                    "Provide detailed information about the cases"

                    "Do not include any disclaimers or unnecessary information."
                    "Do not provide like search strategies and all, you only perform the search and return the results."
                    "Do not provide like As per your request, I have searched the web and all, you only perform the search and return the results."
                    "Do not provide like I was unable to find official links for the cases and citations, you only perform the search and return the results."

                    "At the end provide the link of the some official government websites like ''Casemine', 'sci.gov.in','Drishti Judiciary', 'Delhi Judicial Academy', etc' .",

                    "NOTE : Do not use indiankanoon website for the search."
                    "Provide the SCC citations for all the cases."

                )
            agent_sol = research_agent.run(contents).get_content_as_string()
            
            ilm = gemini.generate_content(
                f"Prompt : {contents}\n Answer the given prompt, additionally you can use the knowledge from agent which is {agent_sol}",
                generation_config=genai.GenerationConfig(
                temperature=0.2,
                )
            )

            cases = genai.GenerativeModel("gemini-2.0-flash").generate_content(f"""
                Provided the information about the case and statute, extract the cases and their citations from the text.
                Give it in a STRICT JSON 
                format, with the following

                "Case Name" : [List of string of cases],
                "Citation" : [List of string of citations],  
                "Necessary Content" : [List of string of necessary content],                                                                                                                                                                    

                The content is {ilm.text}                   
                """).text
            
            dict_ = parse_gemini_response(cases)

            web_links = fetch_official_links(f"The provided is the dictionary of cases and citations {dict_}, provide the official links of the cases using citations or the necessary points given from the Indian government websites liek 'Casemine','sci.gov.in', 'Drishti Judiciary', etc\n Provide the content in a structured format with headings and subheadings.")
            
            st.markdown(ilm.text)
            st.markdown("### Extracted Cases and Citations")
            st.markdown(web_links)
                                        
elif tab == "Text Chat Agent":
    st.header("Text Chat Agent")
    st.subheader("Chat with the Legal Query Chat Agent")

    s_files = st.sidebar.file_uploader("Upload your PDF files",help = "You should be uploading more than one file", type = ['pdf'], accept_multiple_files = True)
    s_files_id = ""

    if "doc_messages" not in st.session_state:
        st.session_state.doc_messages = []

    if "doc_paragraphs" not in st.session_state:
        st.session_state.doc_paragraphs = {}

    if "faiss" not in st.session_state:
        st.session_state.faiss = {}

    for s_file in s_files:
        s_files_id += s_file.file_id
    
    if len(s_files) !=0:
        if st.sidebar.button("Upload file"):
            try:                                        
                texts = ""
                for s_file in s_files:
                    if s_file.file_id not in st.session_state.doc_paragraphs:
                        with st.spinner('Getting the details'):
                            pdf_reader = PdfReader(s_file)
                            text = ''
                            for page in pdf_reader.pages:
                                text+= page.extract_text()
                    
                            st.session_state.doc_paragraphs[s_file.file_id] = text

                    texts+=st.session_state.doc_paragraphs[s_file.file_id]
                
                st.session_state.doc_paragraphs[s_files_id] = texts
                    
                if s_files_id not in st.session_state.faiss:
                    chunks = get_chunks(st.session_state.doc_paragraphs[s_files_id])

                    with st.spinner("Reading records..."):
                        st.session_state.faiss[s_files_id] = get_vector_store(chunks)

                if s_files_id in st.session_state.faiss:
                    st.info("The files are uploaded, you can start the chat now...")
            
            except Exception as e:
                st.error(f"Error Occurred: {e}")

        
        if st.sidebar.button("End Session & Delete Index"):
            index_to_delete = st.session_state.get("user_index_name")
            if index_to_delete:
                with st.spinner(f"Deleting Pinecone index: {index_to_delete}"):
                    try:
                        pc.delete_index(index_to_delete)
                        st.success(f"Deleted index: {index_to_delete}")
                    except Exception as e:
                        st.error(f"Error deleting index: {e}")
                
                for key in [
                    "index", "user_index_name", "doc_paragraphs", "faiss", "doc_messages",
                    "pa_doc_paragraphs", "pa_faiss", "pa_doc_messages", "vector_stores"
                ]:
                    st.session_state.pop(key, None)

            st.rerun()

    h_model = genai.GenerativeModel(model_name= "gemini-2.0-flash", 
    system_instruction = "You are a very professional legal agent related to Indian Laws, and can answer any queries, if the document is uploaded then related to the document in an easier manner and outside the document too"
    )

    doc_chat = st.session_state.doc_messages

    for message in doc_chat:        
        st.chat_message(message['role']).markdown(message['content'])                    

    try:
        user_question = st.chat_input("Enter your query here !!")

        if user_question:
            row_u = st.columns(2)
            row_u[1].chat_message('user').markdown(user_question)
            doc_chat.append(
                {'role': 'user',
                'content': user_question}
            )
        
            if len(s_files) !=0 and st.session_state.doc_paragraphs != {}:
                with st.spinner("Generating response..."):
                    relevant_texts = get_rel_text(user_question, st.session_state.faiss[s_files_id])
                    bot_reply = bot_response(h_model, user_question, relevant_texts, doc_chat)

            else:
                with st.spinner("Generating response..."):  
                    contents=(
                            f"For the given query '{user_question}'",
                            f"Answer the queries like a professional person being in the domain of LEGAL Authority of INDIAN LAW, having a lot of knowledge on the LAW and LEGISLATION of the INDIAN GOVERNMENT",                                
                            f"Search for the latest and most relevant information.",
                            f"Please try to provide the lists of the cases and citations from the official government website 'SCC ONline', 'Manupatra', 'Casemine','sci.gov.in'",
                            f"Please go through this website for the cases and statutes: https://www.aironline.in/index.html, you can also search the cases and case information from the web and using your knowledge also",
                            
                            "For the respoonse Do not include any disclaimers or unnecessary information."
                            "Do not provide like search strategies and all, you only perform the search and return the results."
                            "Do not provide like As per your request, I have searched the web and all, you only perform the search and return the results."
                            "Do not provide like Okay assistant, I have searched the web and all, you only perform the search and return the results."

                            f"Consider the following conversation history: {doc_chat}",

                            "NOTE : Do not use indiankanoon website for the search."
                            "For premium websites, or the websites which are not free, do not provide the links for citations, or provide the citations, like SCC citation, manupatra citation and all",      
                        )

                    agent_sol = research_agent.run(contents).get_content_as_string()                      
                    bot_reply = h_model.generate_content(
                        f"Prompt : {contents}\n Answer the given prompt, additionally you can use the knowledge from agent which is {agent_sol}",                                                   
                    ).text

                    if bot_reply:
                        contr = genai.GenerativeModel("gemini-2.0-flash").generate_content(f"""
                        Provided the information about the case and statute, law or any information regarding the legal domain extract the cases and their citations, or any important keywords and statement from the text.
                        Give it in a STRICT JSON 
                        format, with the following

                        "Case Name" : [List of string of cases],
                        "Citation" : [List of string of citations],   
                        "Necessary Content" : [List of string of necessary content],                                                                                                                                                                                                                                                                      

                        The content is {bot_reply}                                           

                        """).text
                        
                        dict_ = parse_gemini_response(contr)
                        links = fetch_official_links(f"The provided is the dictionary of cases or necessary content and citations {dict_}, provide the official links of the cases using citations or the necessary points given from the Indian government website 'SCC ONline', 'Manupatra', 'Casemine','sci.gov.in'")
                        bot_reply = f"{bot_reply}\n\n---\n**Official Sources:**\n{links}"                                                    

            
            st.chat_message('assistant').markdown(bot_reply)

            doc_chat.append(
                {'role': 'assistant',
                'content': bot_reply}
            )

    except Exception as e:
        st.chat_message('assistant').markdown(f'There might be an error, try again, {str(e)}')
        doc_chat.append(
            {
                'role': 'assistant',
                'content': f'There might be an error, try again, {str(e)}'
            }
        )

elif tab == "Legal Advocate Agent":
    st.header("Legal Advocate Agent")     
    st.subheader("Chat with the Legal Advocate Agent")   

    pa_files = st.sidebar.file_uploader("Upload your PDF files",help = "You should be uploading more than one file", type = ['pdf'], accept_multiple_files = True)
    pa_files_id = ""

    if "pa_doc_messages" not in st.session_state:
        st.session_state.pa_doc_messages = []
        

    if "pa_doc_paragraphs" not in st.session_state:
        st.session_state.pa_doc_paragraphs = {}
        

    if "pa_faiss" not in st.session_state:
        st.session_state.pa_faiss = {}
        

    for pa_file in pa_files:
        pa_files_id += pa_file.file_id
    
    if len(pa_files) !=0:
        if st.sidebar.button("Upload file"):
            try:                                        
                texts = ""
                for pa_file in pa_files:
                    if pa_file.file_id not in st.session_state.pa_doc_paragraphs:
                        with st.spinner('Getting the details'):
                            pdf_reader = PdfReader(pa_file)
                            text = ''
                            for page in pdf_reader.pages:
                                text+= page.extract_text()
                    
                            st.session_state.pa_doc_paragraphs[pa_file.file_id] = text

                    texts+=st.session_state.pa_doc_paragraphs[pa_file.file_id]
                
                st.session_state.pa_doc_paragraphs[pa_files_id] = texts
                    
                if pa_files_id not in st.session_state.pa_faiss:
                    chunks = get_chunks(st.session_state.pa_doc_paragraphs[pa_files_id])

                    with st.spinner("Reading records..."):
                        st.session_state.pa_faiss[pa_files_id] = get_vector_store(chunks)

                if pa_files_id in st.session_state.pa_faiss:
                    st.info("The files are uploaded, you can start the chat now...")
            
            except Exception as e:
                st.error(f"Error Occurred: {e}")


        if st.sidebar.button("End Session & Delete Index"):
            index_to_delete = st.session_state.get("user_index_name")
            if index_to_delete:
                with st.spinner(f"Deleting Pinecone index: {index_to_delete}"):
                    try:
                        pc.delete_index(index_to_delete)
                        st.success(f"Deleted index: {index_to_delete}")
                    except Exception as e:
                        st.error(f"Error deleting index: {e}")
                
                for key in [
                    "index", "user_index_name", "doc_paragraphs", "faiss", "doc_messages",
                    "pa_doc_paragraphs", "pa_faiss", "pa_doc_messages", "vector_stores"
                ]:
                    st.session_state.pop(key, None)

            st.rerun()



    pa_model = genai.GenerativeModel(model_name= "gemini-2.0-flash", 
                                        
    system_instruction = (
        "You are a highly professional legal AI agent specializing in Indian law. "
        "You can answer any legal query with precision, drawing on both uploaded documents and external sources. "
        "When a document is provided, integrate its contents seamlessly into your analysis, while also leveraging your broader knowledge of statutes, IPC sections, and landmark judgments. "
        "Always prioritize clarity, accuracy, and up-to-date information. "
        "Structure responses in a clear, organized manner with sections for Facts & Issues, Relevant Statutes/IPC Sections, Precedents, Legal Analysis, Conclusion, and References. "
        "Cite all cases (with full citations and URLs to 'SCC ONline', 'Manupatra', 'Casemine','sci.gov.in') and statutes used. "
        "Avoid disclaimers, personal opinions, or unnecessary commentary. "
        "If no document is uploaded, rely on authoritative external sources and your legal expertise to address the user’s query comprehensively."

        "NOTE : Do not use indiankanoon website for the search."
        "For premium websites, or the websites which are not free, do not provide the links for citations, or provide the citations, like SCC citation, manupatra citation and all",      
        )

    )

    pa_doc_chat = st.session_state.pa_doc_messages

    for message in pa_doc_chat:        
        st.chat_message(message['role']).markdown(message['content'])                    

    try:
        user_question = st.chat_input("Enter your query here !!")

        if user_question:
            row_u = st.columns(2)
            row_u[1].chat_message('user').markdown(user_question)
            pa_doc_chat.append(
                {'role': 'user',
                'content': user_question}
            )
        
            if len(pa_files) !=0 and st.session_state.pa_doc_paragraphs != {}:
                with st.spinner("Generating response..."):
                    relevant_texts = get_rel_text(user_question, st.session_state.pa_faiss[pa_files_id])
                    bot_reply = agent_bot_response(pa_model, user_question, relevant_texts, pa_doc_chat)

            else:
                with st.spinner("Generating response..."):  
                    contents = (
                        f"For the given query '{user_question}', analyze the issue under Indian law using a Retrieval-Augmented Generation (RAG) approach.",
                        f"Answer the query as a professional legal authority on Indian law, drawing on statutes, IPC sections, and landmark judgments.",
                        "Search for the latest and most relevant information to ensure currency. Prioritize AIR Online (https://www.aironline.in/index.html) for reported judgments.",
                        "Provide a list of relevant cases with full citations (case name, citation number, court, year) and direct URLs to 'SCC ONline', 'Manupatra', 'Casemine','sci.gov.in' wherever available.",
                        "Structure your response in Structured fomrat with the following headings and subheadings:",
                        "  Facts & Issues",
                        "  Relevant Statutes & IPC Sections",
                        "  Precedent Summaries (parties, facts, issues, reasoning, outcome, citation)",
                        "  Legal Analysis & Argument",
                        "  Conclusion",
                        "  References",
                        "Do not include any disclaimers, personal opinions, or unrelated commentary. Focus strictly on legal analysis and citations.",
                        "Do not describe your search strategy or mention tool usage—only present the findings.",
                        f"Consider the following conversation history: {pa_doc_chat}"

                        "NOTE : Do not use indiankanoon website for the search."
                        "For premium websites, or the websites which are not free, do not provide the links for citations, or provide the citations, like SCC citation, manupatra citation and all",      
                    )

                    agent_sol = lawer_agent.run(contents).get_content_as_string()                        
                    bot_reply = pa_model.generate_content(
                        f"Prompt : {contents}\n Answer the given prompt, additionally you can use the knowledge from agent which is {agent_sol}",                                                   
                    ).text

                    if bot_reply:
                        contr = genai.GenerativeModel("gemini-2.0-flash").generate_content(f"""
                        Provided the information about the case and statute, law or any information regarding the legal domain extract the cases and their citations, or any important keywords and statement from the text.
                        Give it in a STRICT JSON 
                        format, with the following

                        "Case Name" : [List of string of cases],
                        "Citation" : [List of string of citations],   
                        "Necessary Content" : [List of string of necessary content],                                                                                                                                                                                                                                                                      

                        The content is {bot_reply}                                                                                                                                                                              

                        """).text
                        
                        dict_ = parse_gemini_response(contr)
                        links = fetch_official_links(f"The provided is the dictionary of cases or necessary content and citations {dict_}, provide the official links of the cases using citations or the necessary points given from the Indian government website 'SCC ONline', 'Manupatra', 'Casemine','sci.gov.in', mostly use 'CaseMine' for the search")
                        bot_reply = f"{bot_reply}\n\n---\n**Official Sources:**\n{links}"                                                    
            
            st.chat_message('assistant').markdown(bot_reply)

            pa_doc_chat.append(
                {'role': 'assistant',
                'content': bot_reply}
            )

    except Exception as e:
        st.chat_message('assistant').markdown(f'There might be an error, try again, {str(e)}')
        pa_doc_chat.append(
            {
                'role': 'assistant',
                'content': f'There might be an error, try again, {str(e)}'
            }
        )

else:
    st.header("Research Paper Gatherer Agent")
    st.subheader("Chat with the Research Paper Gatherer Agent")
    
    rp_files = st.sidebar.file_uploader(
        "Upload Base Legal Papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more base papers for RAG indexing."
    )

    if "rp_doc_messages" not in st.session_state:
        st.session_state.rp_doc_messages = []
    if "rp_doc_paragraphs" not in st.session_state:
        st.session_state.rp_doc_paragraphs = {}
    if "rp_vector_store" not in st.session_state:
        st.session_state.rp_vector_store = None
   
    index_name = st.session_state.user_index_name
    
    if rp_files and st.sidebar.button("Upload Base Papers"):
        try:
            concatenated_text = ""
            for rp_file in rp_files:
                if rp_file.file_id not in st.session_state.rp_doc_paragraphs:
                    with st.spinner(f"Parsing {rp_file.name}..."):
                        reader = PdfReader(rp_file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() or ""
                    st.session_state.rp_doc_paragraphs[rp_file.file_id] = text

                concatenated_text += st.session_state.rp_doc_paragraphs[rp_file.file_id]
            
            chunks = get_chunks(concatenated_text)
            with st.spinner("Building Pinecone vector index..."):
                st.session_state.rp_vector_store = get_vector_store(chunks)

            st.success(f"Indexed Base Papers: {', '.join(f.name for f in rp_files)}")
        except Exception as e:
            st.error(f"Error during indexing: {e}")
    
    if st.sidebar.button("Clear Papers & Reset"):        
        try:
            del_index(index_name)
        except Exception:
            pass
        for key in ["rp_doc_messages", "rp_doc_paragraphs", "rp_vector_store"]:
            st.session_state.pop(key, None)
        st.rerun()

    rp_messages = st.session_state.rp_doc_messages
    for message in rp_messages:        
        st.chat_message(message['role']).markdown(message['content'])                    

    try:
        user_query = st.chat_input("Enter your research query (e.g., 'Draft an outline for a paper on digital evidence')")

        if user_query:
            cols_u = st.columns(2)
            cols_u[1].chat_message("user").markdown(user_query)
            rp_messages.append({"role": "user", "content": user_query})

            if st.session_state.rp_vector_store:
                with st.spinner("Retrieving relevant sections from base paper..."):
                    relevant_chunk = get_rel_text(user_query, st.session_state.rp_vector_store)
                with st.spinner("Generating response..."):
                    bot_response = research_paper_agent_response(
                        research_paper_agent, user_query, relevant_chunk, rp_messages
                    )
            else:                
                with st.spinner("Generating response (no base paper)..."):
                    bot_response = research_paper_agent_response(
                        research_paper_agent, user_query, "", rp_messages
                    )
           
            st.chat_message("assistant").markdown(bot_response)
            rp_messages.append({"role": "assistant", "content": bot_response})

    except Exception as e:
        st.chat_message("assistant").markdown(f"Error generating response: {e}")
        rp_messages.append({"role": "assistant", "content": f"Error: The files are not uploaded"})