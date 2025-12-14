import os
import io
import uuid
import json
import re
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from PyPDF2 import PdfReader

import google.generativeai as genai
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.googlesearch import GoogleSearch
from phi.tools.website import WebsiteTools
from phi.tools.wikipedia import WikipediaTools

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# -------------------------------
# Environment and global models
# -------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def _new_index_name() -> str:
    return f"legal-qa-db-{uuid.uuid4().hex[:8]}"


def _create_index() -> Dict[str, Any]:
    index_name = _new_index_name()
    if index_name not in [i['name'] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1'),
        )
    return {"index_name": index_name, "index": pc.Index(index_name)}


def _delete_index(index_name: str) -> None:
    try:
        pc.delete_index(index_name)
    except Exception:
        pass


def get_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text or "")


def get_vector_store(text_chunks: List[str], index) -> PineconeVectorStore:
    vs = PineconeVectorStore(index=index, embedding=embeddings)
    if text_chunks:
        vs.add_texts(text_chunks)
    return vs


def get_rel_text(user_question: str, db: PineconeVectorStore) -> str:
    docs = db.similarity_search(user_question, k=1)
    return docs[0].page_content if docs else ""


def parse_gemini_response(response_text: str):
    try:
        cleaned = re.sub(r'```json|```', '', response_text or "")
        json_str = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_str:
            return json.loads(json_str.group())
        return None
    except Exception:
        return None


# -------------------------------
# Agents (mirroring main.py behavior, but UI-agnostic)
# -------------------------------
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
        "NOTE : Do not use indiankanoon website for the search.",
        "Provide the SCC citations for all the cases.",
        "Search through the web for citation of cases and all",
    ],
    markdown=True,
    show_tool_calls=False,
)

website_agent = Agent(
    name="Legal Website Agent",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[WebsiteTools(), GoogleSearch()],
    description="Fetch official government/legal website links and PDF documents.",
    instructions=[
        "Provide official links for cases and citations.",
        "Search for PDF documents from the web",
        "Provide the citations and links for the cases and statutes.",
        "Try to provide the links of cases from Casemine, and other official websites like 'Drishti Judiciary, Delhi Judicial Academy, etc'",
        "Search for the latest and most relevant information through the web and using your knowledge.",
        "Provide the available SCC citations for all the cases.",
        "NOTE : Do not use indiankanoon website for the search.",
        "Do not provide hallucinations or irrelevant information.",
        "Search through the web for citation of cases and all",
        "Do not provide like search strategies and all, you only perform the search and return the results.",
    ],
    markdown=True,
    show_tool_calls=False,
)

lawer_agent = Agent(
    name="Lawyer PA",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[GoogleSearch(), WebsiteTools(), WikipediaTools()],
    description=(
        "You are a professional legal AI agent specialized in Indian law, trained on Phi data and capable of researching, analyzing, and arguing hypothetical cases."
    ),
    instructions=[
        "When a user presents a hypothetical or real case fact pattern, identify the key legal issues and relevant facts.",
        "Search for analogous Indian court decisions, statutes, and IPC sections using provided tools (GoogleSearch, WebsiteTools, WikipediaTools).",
        "For each relevant precedent, extract: parties, facts, legal issues, reasoning, outcome, and citations (with official links). Summarize concisely with headings.",
        "Organize the research report into clear sections: (a) Facts & Issues; (b) Relevant Statutes/IPC Sections; (c) Precedent Summaries; (d) Argument Outline; (e) Possible Counterarguments; (f) Conclusion & Recommendations.",
        "Based on analogous cases, construct logical arguments for the hypothetical scenario—identifying strengths, weaknesses, and potential defenses—while citing statutes and case law.",
        "Always include full citations and direct URLs to primary sources in a References section.",
        "Deliver the final output in Markdown format with nested headings, bullet points, and numbered lists for readability.",
        "Avoid adding disclaimers, personal opinions, or unrelated commentary. Focus strictly on legal analysis.",
        "NOTE : Do not use indiankanoon website for the search.",
        "Provide the SCC citations for all the cases.",
        "Search through the web for citation of cases and all",
    ],
    markdown=True,
    show_tool_calls=False,
)

research_paper_agent = Agent(
    name="Legal Research Paper Agent",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[GoogleSearch(), WebsiteTools(), WikipediaTools()],
    description=(
        "You are an AI agent specialized in gathering, summarizing, and synthesizing legal research papers and journal articles."
    ),
    markdown=True,
    show_tool_calls=False,
)


def fetch_official_links(query: str) -> str:
    try:
        prompt = (
            f"Given the dictionary {query} consisting of the cases or citations, search for official links (Casemine, sci.gov.in, etc.). Also provide available SCC citations."
        )
        result = website_agent.run(prompt)
        return result.get_content_as_string()
    except Exception as e:
        return f"_Error fetching official links: {e}_"


def bot_response(model, query: str, relevant_text: str, history: List[Dict[str, str]]):
    context = ' '.join([relevant_text]) if isinstance(relevant_text, str) else ' '.join(relevant_text)
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

    Please try to provide the lists of the cases and citations from the official government website 'Casemine', 'sci.gov.in', 'Drishti Judiciary', 'Delhi Judicial Academy', etc.
    """

    agent_sol = research_agent.run(prompt).get_content_as_string()
    response = model.generate_content(
        f"Prompt : {prompt}\n Answer the given prompt, additionally you can use the knowledge from agent which is {agent_sol}",
        generation_config=genai.GenerationConfig(temperature=0.2),
    ).text

    if response:
        contr = genai.GenerativeModel("gemini-2.0-flash").generate_content(
            f"""
            Provided the information about the case and statute, law or any information regarding the legal domain extract the cases and their citations, or any important keywords and statement from the text.
            Give it in a STRICT JSON format with keys: "Case Name", "Citation", "Necessary Content".
            The content is {response}
            """
        ).text
        dict_ = parse_gemini_response(contr)
        links = fetch_official_links(f"The provided is the dictionary of cases and citations {dict_}")
        response = f"{response}\n\n---\n**Official Sources:**\n{links}"

    return response


def agent_bot_response(model, query: str, relevant_text: str, history: List[Dict[str, str]]):
    context = ' '.join([relevant_text]) if isinstance(relevant_text, str) else ' '.join(relevant_text)
    prompt = f"""This is the context of the document:
        Context: {context}
        This is the user query:
        User: {query}
        This is the history of the conversation:
        History: {history}
        YOU ARE A LEGAL RESEARCH ASSISTANT SPECIALIZING IN INDIAN LAW. Provide accurate, reliable, and contextual answers with citations and official links.
        """

    agent_sol = lawer_agent.run(prompt).get_content_as_string()
    response = model.generate_content(
        f"Prompt : {prompt}\n Answer the given prompt, additionally you can use the knowledge from agent which is {agent_sol}",
        generation_config=genai.GenerationConfig(temperature=0.2),
    ).text

    if response:
        contr = genai.GenerativeModel("gemini-2.0-flash").generate_content(
            f"""
            Provided the information about the case and statute, law or any information regarding the legal domain extract the cases and their citations, or any important keywords and statement from the text.
            Give it in a STRICT JSON format with keys: "Case Name", "Citation", "Necessary Content".
            The content is {response}
            """
        ).text
        dict_ = parse_gemini_response(contr)
        links = fetch_official_links(f"The provided is the dictionary of cases and citations {dict_}")
        response = f"{response}\n\n---\n**Official Sources:**\n{links}"

    return response


# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI(title="Indian Legal Research Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SESSIONS: Dict[str, Dict[str, Any]] = {}


class SearchRequest(BaseModel):
    query: str


class ChatMessageRequest(BaseModel):
    session_id: str
    question: str


class StartSessionResponse(BaseModel):
    session_id: str


class ResearchQueryRequest(BaseModel):
    session_id: str
    query: str


@app.get("/")
def root():
    return {"message": "Indian Legal Research Agent FastAPI is running."}


@app.post("/search")
def search_hub(req: SearchRequest):
    contents = (
        f"For the given query '{req.query}'",
        f"Provide relevant Indian statutes, citation and details, IPC sections, landmark cases",
        "If the query is not related to Indian law, please say 'Not related to Indian law'.",
        "If the query is of a particular case, include that case's name and citation.",
        "Provide the results in markdown format with headings and subheadings.",
        "Do not provide hypothetical examples or case studies.",
        "Search for the latest and most relevant information.",
        "Please go through this website for the cases and statutes: https://www.aironline.in/index.html",
        "You can also search cases from the web and using your knowledge as well.",
        "Do not include any disclaimers or unnecessary information.",
        "NOTE: Do not use indiankanoon website for the search.",
        "Provide the SCC citations for all the cases.",
    )

    agent_sol = research_agent.run(contents).get_content_as_string()
    gem = genai.GenerativeModel("gemini-2.0-flash")
    ilm = gem.generate_content(
        f"Prompt : {contents}\n Answer the given prompt, additionally you can use the knowledge from agent which is {agent_sol}",
        generation_config=genai.GenerationConfig(temperature=0.2),
    )

    cases = genai.GenerativeModel("gemini-2.0-flash").generate_content(
        f"""
        Provided the information about the case and statute, extract the cases and their citations from the text.
        Give it in a STRICT JSON with keys: "Case Name", "Citation", "Necessary Content".
        The content is {ilm.text}
        """
    ).text

    dict_ = parse_gemini_response(cases)
    web_links = fetch_official_links(
        f"The provided is the dictionary of cases and citations {dict_}, provide the official links of the cases using citations or the necessary points given from the Indian government websites like 'Casemine','sci.gov.in', 'Drishti Judiciary', etc. Provide the content in a structured format with headings and subheadings."
    )
    return {"result": ilm.text, "official_links": web_links}


# -------------------------------
# Text Chat session
# -------------------------------
@app.post("/text/start", response_model=StartSessionResponse)
def start_text_session():
    session_id = uuid.uuid4().hex
    idx = _create_index()
    SESSIONS[session_id] = {
        "type": "text",
        "messages": [],
        "vector_store": {},
        "index_name": idx["index_name"],
        "index": idx["index"],
    }
    return StartSessionResponse(session_id=session_id)


@app.post("/text/upload")
async def upload_text_pdfs(session_id: str = Form(...), files: List[UploadFile] = File(...)):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    texts = ""
    for f in files:
        content = await f.read()
        try:
            reader = PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception:
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {f.filename}")
        texts += text

    chunks = get_chunks(texts)
    session["vector_store"]["combined"] = get_vector_store(chunks, session["index"])
    return {"message": "Files uploaded and indexed."}


@app.post("/text/message")
def text_message(req: ChatMessageRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    h_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=(
            "You are a very professional legal agent related to Indian Laws. If documents are uploaded, answer using them as context; you can also answer outside the document."
        ),
    )

    messages = session["messages"]
    messages.append({"role": "user", "content": req.question})

    if session["vector_store"]:
        relevant_texts = get_rel_text(req.question, session["vector_store"]["combined"])  # str
        bot_reply = bot_response(h_model, req.question, relevant_texts, messages)
    else:
        contents = (
            f"For the given query '{req.question}'",
            "Answer the queries like a professional person being in the domain of LEGAL Authority of INDIAN LAW, having a lot of knowledge on the LAW and LEGISLATION of the INDIAN GOVERNMENT",
            "Search for the latest and most relevant information.",
            "Please try to provide the lists of the cases and citations from 'SCC ONline', 'Manupatra', 'Casemine','sci.gov.in'",
            "Please go through this website for the cases and statutes: https://www.aironline.in/index.html",
        )
        agent_sol = research_agent.run(contents).get_content_as_string()
        bot_reply = h_model.generate_content(
            f"Prompt : {contents}\n Answer the given prompt, additionally you can use the knowledge from agent which is {agent_sol}",
        ).text
        if bot_reply:
            contr = genai.GenerativeModel("gemini-2.0-flash").generate_content(
                f"""
                Provided the information about the legal domain extract the cases and their citations, or important keywords.
                Strict JSON with keys: "Case Name", "Citation", "Necessary Content".
                The content is {bot_reply}
                """
            ).text
            dict_ = parse_gemini_response(contr)
            links = fetch_official_links(
                f"The provided is the dictionary of cases or necessary content and citations {dict_}"
            )
            bot_reply = f"{bot_reply}\n\n---\n**Official Sources:**\n{links}"

    messages.append({"role": "assistant", "content": bot_reply})
    return {"answer": bot_reply}


@app.delete("/text/end")
def end_text_session(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    index_name = session.get("index_name")
    if index_name:
        _delete_index(index_name)
    SESSIONS.pop(session_id, None)
    return {"message": "Session ended and index deleted."}


# -------------------------------
# Advocate session
# -------------------------------
@app.post("/advocate/start", response_model=StartSessionResponse)
def start_advocate_session():
    session_id = uuid.uuid4().hex
    idx = _create_index()
    SESSIONS[session_id] = {
        "type": "advocate",
        "messages": [],
        "vector_store": {},
        "index_name": idx["index_name"],
        "index": idx["index"],
    }
    return StartSessionResponse(session_id=session_id)


@app.post("/advocate/upload")
async def upload_advocate_pdfs(session_id: str = Form(...), files: List[UploadFile] = File(...)):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    texts = ""
    for f in files:
        content = await f.read()
        try:
            reader = PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception:
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {f.filename}")
        texts += text

    chunks = get_chunks(texts)
    session["vector_store"]["combined"] = get_vector_store(chunks, session["index"])
    return {"message": "Files uploaded and indexed."}


@app.post("/advocate/message")
def advocate_message(req: ChatMessageRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    pa_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=(
            "You are a highly professional legal AI agent specializing in Indian law. Structure responses with sections and include citations and links."
        ),
    )

    messages = session["messages"]
    messages.append({"role": "user", "content": req.question})

    if session["vector_store"]:
        relevant_texts = get_rel_text(req.question, session["vector_store"]["combined"])  # str
        bot_reply = agent_bot_response(pa_model, req.question, relevant_texts, messages)
    else:
        contents = (
            f"For the given query '{req.question}', analyze the issue under Indian law using a RAG approach.",
            f"Answer as a professional legal authority on Indian law, citing statutes, IPC sections, landmark judgments.",
            "Search for the latest and most relevant information. Prioritize AIR Online (https://www.aironline.in/index.html).",
        )
        agent_sol = lawer_agent.run(contents).get_content_as_string()
        bot_reply = pa_model.generate_content(
            f"Prompt : {contents}\n Answer the given prompt, additionally you can use the knowledge from agent which is {agent_sol}",
        ).text
        if bot_reply:
            contr = genai.GenerativeModel("gemini-2.0-flash").generate_content(
                f"""
                Provided the information extract the cases and their citations.
                Strict JSON with keys: "Case Name", "Citation", "Necessary Content".
                The content is {bot_reply}
                """
            ).text
            dict_ = parse_gemini_response(contr)
            links = fetch_official_links(
                f"The provided is the dictionary of cases or necessary content and citations {dict_}"
            )
            bot_reply = f"{bot_reply}\n\n---\n**Official Sources:**\n{links}"

    messages.append({"role": "assistant", "content": bot_reply})
    return {"answer": bot_reply}


@app.delete("/advocate/end")
def end_advocate_session(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    index_name = session.get("index_name")
    if index_name:
        _delete_index(index_name)
    SESSIONS.pop(session_id, None)
    return {"message": "Session ended and index deleted."}


# -------------------------------
# Research Papers session
# -------------------------------
@app.post("/papers/start", response_model=StartSessionResponse)
def start_papers_session():
    session_id = uuid.uuid4().hex
    idx = _create_index()
    SESSIONS[session_id] = {
        "type": "papers",
        "messages": [],
        "vector_store": None,
        "index_name": idx["index_name"],
        "index": idx["index"],
    }
    return StartSessionResponse(session_id=session_id)


@app.post("/papers/upload")
async def upload_papers(session_id: str = Form(...), files: List[UploadFile] = File(...)):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    concatenated_text = ""
    for f in files:
        content = await f.read()
        try:
            reader = PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception:
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {f.filename}")
        concatenated_text += text

    chunks = get_chunks(concatenated_text)
    session["vector_store"] = get_vector_store(chunks, session["index"])
    return {"message": "Base papers indexed."}


@app.post("/papers/message")
def papers_message(req: ResearchQueryRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = session["messages"]
    messages.append({"role": "user", "content": req.query})

    if session["vector_store"]:
        relevant_chunk = get_rel_text(req.query, session["vector_store"])  # str
        bot_resp = research_paper_agent.run(
            f"Context: {relevant_chunk}\nUser Query: {req.query} give it in html tags"
        ).get_content_as_string()
    else:
        bot_resp = research_paper_agent.run(
            f"User Query: {req.query}"
        ).get_content_as_string()

    messages.append({"role": "assistant", "content": bot_resp})
    return {"answer": bot_resp}


@app.delete("/papers/reset")
def reset_papers(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    index_name = session.get("index_name")
    if index_name:
        _delete_index(index_name)
    SESSIONS.pop(session_id, None)
    return {"message": "Papers session cleared and index deleted."}


