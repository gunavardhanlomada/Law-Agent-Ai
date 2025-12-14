<h2>Setup & Installation</h2>

<h3>Clone Repository</h3>
<pre>
git clone https://github.com/gunavardhanlomada/Law-Agent-Ai.git
cd Law-Agent-Ai
</pre>

<h3>Install Dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<h3>Environment Variables</h3>
<pre>
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
SERP_API_KEY=your_serp_api_key
</pre>

<h3>Run Server</h3>
<pre>
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
</pre>

<hr/>

<h2>📌 Example Use Case</h2>
<p>
Upload a legal agreement and ask:
<b>“Can a builder execute sale deeds without landowner presence in Telangana?”</b>
</p>

<p>
The system returns:
</p>
<ul>
  <li>Applicable statutes</li>
  <li>SCC-cited judgments</li>
  <li>Structured legal reasoning</li>
  <li>Official source links</li>
</ul>
<hr/>
Note: This app avoids unofficial sources like Indian Kanoon and focuses on government-approved databases.
