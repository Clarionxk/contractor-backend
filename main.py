import os
import re
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from typing import Optional
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool


# -------------------
# Load environment
# -------------------
load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not serper_api_key or not openai_api_key:
    raise ValueError("API keys are missing. Please check your .env file.")

os.environ["SERPER_API_KEY"] = serper_api_key

# -------------------
# Initialize LLM + tool
# -------------------
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key
)
search_tool = SerperDevTool()

# -------------------
# Define Agents
# -------------------
law_expert = Agent(
    role='Global Legal Scholar and Analyst',
    goal="Perform comprehensive legal analysis for contracts across different jurisdictions.",
    backstory="With decades of experience in legal academia and practice...",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[search_tool]
)

contract_connoisseur = Agent(
    role='Master Contract Draftsman and Linguistic Expert',
    goal="Draft impeccable contracts that are legally binding and contextually appropriate.",
    backstory="You are an elite contract lawyer with mastery of legal drafting...",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[search_tool]
)

proofreader = Agent(
    role='Contract Quality Assurance Specialist and Compliance Auditor',
    goal="Perform in-depth reviews of drafted contracts for quality and legal compliance.",
    backstory="With a background in legal compliance and quality assurance...",
    allow_delegation=True,
    llm=llm,
    tools=[search_tool]
)

# -------------------
# FastAPI Setup
# -------------------
app = FastAPI(title="Contract Generator Backend", version="1.0.0")

WP_ORIGIN = os.getenv("WP_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[WP_ORIGIN] if WP_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Helper
# -------------------
def build_html_response(legal: str, draft: str, review: str, logs: str) -> str:
    return f"""
<h2>Legal Requirements Output</h2>
<pre>{legal}</pre>
<h2>Draft Contract Output</h2>
<pre>{draft}</pre>
<h2>Reviewed Contract Output</h2>
<pre>{review}</pre>
<h2>Agent Process Logs</h2>
<pre>{logs}</pre>
""".strip()

# -------------------
# Routes
# -------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return "FastAPI backend is running on Render!"

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"



CATEGORIES = {
    "business contracts": [
        "Sales Contract",
        "Partnership Agreement",
        "Service Agreement",
        "Independent Contractor Agreement",
        "Non-Disclosure Agreement",
    ],
    "employment contracts": [
        "Employment Agreement",
        "Offer Letter",
        "NDA (Mutual)",
        "Consulting Agreement",
        "Non-Compete Agreement",
    ],
}

@app.get("/get-contract-types", response_class=HTMLResponse)
def get_contract_types(
    contract_category: Optional[str] = Query(None, alias="contract-category"),
    category: Optional[str] = None,
):
    # accept both ?contract-category=... and ?contract_category=... or ?category=...
    raw = contract_category or category
    selected = (raw or "Business Contracts").strip().lower()

    types = CATEGORIES.get(selected, CATEGORIES["business contracts"])
    html_options = "".join(f"<option value=\"{t}\">{t}</option>" for t in types)
    return HTMLResponse(content=html_options, status_code=200)





@app.post("/generate", response_class=HTMLResponse)
async def generate_contract(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    country = data.get("country", "")
    category = data.get("contract_category", "")
    ctype = data.get("contract_type", "")

    if not (prompt and country and category and ctype):
        return HTMLResponse(
            "<p>Missing required fields (prompt, country, contract_category, contract_type).</p>",
            status_code=400,
        )

    # -------------------
    # CrewAI Pipeline
    # -------------------
    legal_task = Task(
        description=f"List legal requirements for {ctype} in {country}.",
        agent=law_expert,
        expected_output="A bullet list of legal requirements."
    )
    draft_task = Task(
        description=f"Draft a full {ctype} for {country}, based on user prompt:\n{prompt}",
        agent=contract_connoisseur,
        expected_output="A full draft contract."
    )
    review_task = Task(
        description="Review and refine the draft, adding compliance notes.",
        agent=proofreader,
        expected_output="Reviewed contract + compliance notes."
    )

    crew = Crew(
        agents=[law_expert, contract_connoisseur, proofreader],
        tasks=[legal_task, draft_task, review_task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        result = crew.kickoff(inputs={"prompt": prompt})
        outputs = {"legal": "", "draft": "", "review": "", "logs": ""}

        tos = getattr(result, "tasks_output", None)
        if isinstance(tos, list):
            for i, t in enumerate(tos):
                raw = getattr(t, "raw", None) or str(t)
                if i == 0: outputs["legal"] = raw
                elif i == 1: outputs["draft"] = raw
                elif i == 2: outputs["review"] = raw
            outputs["logs"] = str(result)

        if not outputs["review"]:
            outputs["review"] = getattr(result, "raw", str(result))

        return build_html_response(
            outputs["legal"], outputs["draft"], outputs["review"], outputs["logs"]
        )
    except Exception as e:
        logging.exception("Pipeline failed: %s", e)
        return build_html_response(
            f"(Error) {e}", f"(Prompt) {prompt}", "(No review)", "Pipeline crashed"
        )
