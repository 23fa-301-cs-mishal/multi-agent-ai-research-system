import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import certifi
# if you use windows youll get path related error so to fix it we add these 2 lines bellow
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()

# from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph,START,MessagesState
from langgraph.prebuilt import tools_condition,ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from agent_tools import tools
from langchain_openai import ChatOpenAI
#create directory 
Path("data").mkdir(exist_ok=True)

load_dotenv()

DEFAULT_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3.5-super-vl:free"
)

# Active, non-deprecated FREE models on OpenRouter with 1M+ context & tool support
ALLOWED_MODELS = {
    "google/gemma-4-26b-a4b-it:free",
    "inclusionai/ling-3.2-flash:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "openrouter/free",
}

# OpenRouter tries these models if the selected model/provider is unavailable
# or rate-limited. Keep this list aligned with ALLOWED_MODELS.
MODEL_FALLBACKS = [
    "particle-ai/particle-1:free",
    "inclusionai/ling-3.2-flash:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "openrouter/free",
]


SYSTEM_PROMPT = """
You are an AI Research Assistant.

You have the following tools:

1. openALex_search:
   Search for academic research papers links using OpenAlex.

2. search_uploaded_documents:
   Retrieve relevant information from uploaded documents.

3. scraper:
  use when you want to  Extract content from a URL.

4. remember_this:
   Save important user information when requested.

5. recall_memory:
   Retrieve previously saved user information.

IMPORTANT RULES:

- For simple greetings, respond naturally without calling tools.
- For general questions, answer using your knowledge.
- For academic research questions, use openALex_search to get URLs
-STRICT DOCUMENT RETRIEVAL RULES

  1.MANDATORY TOOL USE: For EVERY question concerning uploaded documents, you MUST call search_uploaded_documents. NEVER answer from memory or conversation history alone.

  2.LATEST DOCUMENT DEFAULT: ALWAYS retrieve information from the MOST RECENTLY UPLOADED DOCUMENT. This is the default for ALL document-related questions, even if multiple older documents exist.

  3.NEW UPLOAD OVERRIDES OLD: As soon as the user uploads another document, that document IMMEDIATELY replaces the previous document as the default retrieval source. All subsequent document questions refer to the NEWEST upload unless explicitly stated otherwise.

  4.PREVIOUS DOCUMENT EXCEPTION: Retrieve an older document ONLY when the user EXPLICITLY asks about a previous, earlier, older, or specifically identified document. Do NOT choose an older document based on assumptions.

  5.COMPARISON EXCEPTION: When the user requests a comparison, retrieve EVERY document required for that comparison, including the latest and relevant older documents.

FINAL ENFORCEMENT: When the user's reference is not explicitly historical or comparative, ALWAYS choose the LATEST UPLOAD. NEVER default to all documents or an older upload.
LONG-TERM USER MEMORY RULES (MANDATORY):
- When the user states a personal fact or preference about themselves (for example, their name, role, location, goals, or preferences), call remember_this during that turn, even without an explicit request to remember it. Save a concise, standalone statement such as "The user's name is Mina." Do not save transient details or facts about other people unless asked.
- When the user asks about a personal fact or preference that may have been shared earlier (for example, "What is my name?"), call recall_memory before answering. Use the returned memory as the source of truth. If it contains the answer, answer directly; if not, say you could not find it.
- After remember_this succeeds, acknowledge that the fact was saved. Never claim it was saved unless the tool succeeded.
- Use scraper to scrape URL to get and analyze content.
- Never invent research papers, URLs, or citations.
- If a tool fails, explain the error instead of pretending it succeeded.


Tools: `search_web`, `process_urls` (scrapes to DB), **RAG tool (retrieves from DB)**, `remember_this`, `recall_memory`.
"""


def normalize_model_name(model_name:str|None)->str:
    """
    validate selected model from frontend.
    if model is missing or not allowed ,fallback to DEFAULT_MODEL.
    """
    if not model_name:
        return DEFAULT_MODEL
    model_name=model_name.strip()
    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL
    return model_name


def build_agent(model_name:str):
    """
    build one langGraph agent for a selected Gemini model
    """
    selected_model=normalize_model_name(model_name)
    #initialize ChatGroq
    llm = ChatOpenAI(
        model=selected_model,
        api_key=os.getenv("OPENROUTER_API_KEY"), # type: ignore
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
        extra_body={
            "models": [
                fallback for fallback in MODEL_FALLBACKS
                if fallback != selected_model
            ][:3]
        },
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "AI Research Assistant",
        },
    )

    llm_with_tools=llm.bind_tools(tools)
    def chatbot_node(state:MessagesState):
        messages=[SystemMessage(content=SYSTEM_PROMPT)]+state["messages"]
        response=llm_with_tools.invoke(messages)
        return{
            "messages":[response]
        }
    tool_node=ToolNode(tools)
    workflow=StateGraph(MessagesState)
    workflow.add_node("chatbot",chatbot_node)
    workflow.add_node("tools",tool_node)
    workflow.add_edge(START,"chatbot")
    workflow.add_conditional_edges("chatbot",tools_condition)
    workflow.add_edge("tools","chatbot")

# persistent memory
# inside data folder we create a database object langraph_checkpoints
# sqlite is a local database
    conn=sqlite3.connect(
        "data/langraph_checkpoints.sqlite",
        check_same_thread=False
    )
    checkpointer=SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)



_AGENT_CACHE={}
def get_agent(model_name:str | None=None):
    """
    Return cached langraph agent for selected model.
    if not created yet,create it once and resuse it.
    """
    selected_model=normalize_model_name(model_name)
    if selected_model not in _AGENT_CACHE:
      _AGENT_CACHE[selected_model]=build_agent(selected_model)
    return _AGENT_CACHE[selected_model]
    