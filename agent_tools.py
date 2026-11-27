
from dotenv import load_dotenv
from langchain_core.tools import tool

from Database import save_memory, search_memory
from Search import search_research_papers
from normalextract import extract_pdf_text
from Rag import retrieve_from_rag

from academic_rag import (
    add_academic_papers_to_rag,
    retrieve_from_academic_rag,
    get_academic_rag_statistics
)

load_dotenv()


# ---------------------------------------------------------
# 1. CURRENT THREAD
# ---------------------------------------------------------

CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id


# ---------------------------------------------------------
# 2. OPENALEX SEARCH TOOL
# ---------------------------------------------------------

@tool
def openALex_search(topic: str, limit: int = 10):
    """
    Search OpenAlex for research papers based on the user's
    research topic.

    Returns paper metadata and available PDF URLs.

    The user should select which papers they want processed.
    """
    return search_research_papers(
        topic=topic,
        limit=limit
    )


# ---------------------------------------------------------
# 3. SEARCH UPLOADED DOCUMENTS TOOL
# ---------------------------------------------------------

@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search uploaded PDFs, DOCX, TXT files, notes and documents.

    Use this tool when the user asks about their uploaded files.

    This searches the separate uploaded-document RAG collection.
    """
    return retrieve_from_rag(
        query=query,
        thread_id=CURRENT_THREAD_ID
    )


# ---------------------------------------------------------
# 4. EXISTING GENERAL SCRAPER TOOL
# ---------------------------------------------------------

@tool
def scraper(url: str) -> str:
    """
    Extract text from a given PDF URL.

    Uses normalextract.py and removes the references section
    where detected.
    """
    return extract_pdf_text(url)


# ---------------------------------------------------------
# 5. SCRAPE USER-SELECTED RESEARCH PAPERS
# ---------------------------------------------------------

@tool
def scrape_selected_papers(papers: list[dict]) -> dict:
    """
    Scrape only the research papers selected by the user.

    Input:
        Paper dictionaries returned by openALex_search.

    For each selected paper:
        - Tries its available PDF URLs.
        - Extracts PDF text using normalextract.py.
        - Preserves original OpenAlex metadata.
        - Associates extracted text with its metadata.

    Returns:
        scraped_papers: Successfully extracted papers.
        failed_papers: Papers that could not be extracted.

    Pass only scraped_papers to ingest_academic_papers.
    """

    scraped_papers = []
    failed_papers = []

    for paper in papers:

        title = paper.get("title", "Unknown paper")

        # Get available PDF URLs from Search.py.
        pdf_urls = paper.get("pdf_urls") or []

        # Fallback to the main PDF URL.
        if not pdf_urls and paper.get("pdf_url"):
            pdf_urls = [paper["pdf_url"]]

        if isinstance(pdf_urls, str):
            pdf_urls = [pdf_urls]

        # Remove duplicate and empty URLs.
        pdf_urls = list(dict.fromkeys(
            url for url in pdf_urls if url
        ))

        if not pdf_urls:

            failed_papers.append({
                "title": title,
                "reason": "No PDF URL available."
            })

            continue

        extracted_text = None
        successful_url = None
        errors = []

        # Try available URLs until one works.
        for url in pdf_urls:

            try:

                text = extract_pdf_text(url)

                if not isinstance(text, str) or not text.strip():
                    raise ValueError("No readable text extracted.")

                extracted_text = text
                successful_url = url

                break

            except Exception as e:

                errors.append(f"{url}: {str(e)}")

        # If all URLs fail, record the failure.
        if not extracted_text:

            failed_papers.append({
                "title": title,
                "reason": errors
            })

            continue

        # Preserve original OpenAlex metadata.
        metadata = dict(paper)

        # Preserve citation URL separately.
        metadata["citation_pdf_url"] = (
            paper.get("citation_pdf_url")
            or paper.get("pdf_url")
            or successful_url
        )

        # Record the URL that actually provided the text.
        metadata["extraction_pdf_url"] = successful_url

        # Associate extracted text with its metadata.
        scraped_papers.append({
            "content": extracted_text,
            "metadata": metadata
        })

    return {
        "scraped_papers": scraped_papers,
        "failed_papers": failed_papers,
        "scraped_count": len(scraped_papers),
        "failed_count": len(failed_papers)
    }


# ---------------------------------------------------------
# 6. INGEST SCRAPED PAPERS INTO ACADEMIC RAG
# ---------------------------------------------------------

@tool
def ingest_academic_papers(scraped_papers: list[dict]) -> str:
    """
    Store successfully scraped research papers in Academic ChromaDB.

    Input:
        The scraped_papers list returned by scrape_selected_papers.

    Each paper contains:
        - content: Extracted PDF text.
        - metadata: Original OpenAlex paper metadata.

    Academic RAG:
        - Splits the extracted text into chunks.
        - Attaches metadata to every chunk.
        - Generates embeddings.
        - Stores chunks in the existing Academic ChromaDB.

    Already stored papers are skipped to prevent duplicates.
    """

    if not scraped_papers:
        return "No successfully scraped papers were provided for ingestion."

    result = add_academic_papers_to_rag(scraped_papers)

    return str(result)


# ---------------------------------------------------------
# 7. SEARCH ACADEMIC RAG
# ---------------------------------------------------------

@tool
def academic_research_rag(query: str) -> str:
    """
    Retrieve relevant research information from Academic ChromaDB.

    Uses:
        - ANN vector search
        - BM25 keyword retrieval
        - Reciprocal Rank Fusion
        - CrossEncoder reranking

    Returns relevant academic content with source metadata,
    including titles, authors, publication years, DOIs and URLs.

    Searches the existing academic collection, including
    previously stored and newly ingested papers.
    """

    return retrieve_from_academic_rag(
        query=query,
        k=15,
        candidate_k=70
    )


# ---------------------------------------------------------
# 8. ACADEMIC RAG STATISTICS
# ---------------------------------------------------------

@tool
def academic_rag_statistics() -> str:
    """
    Return Academic ChromaDB statistics, including total
    stored chunks and unique research papers.
    """

    return str(get_academic_rag_statistics())


# ---------------------------------------------------------
# 9. REMEMBER USER INFORMATION
# ---------------------------------------------------------

@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact into long-term memory.

    Use this whenever the user shares a durable personal fact or preference, even without an explicit request to remember it.
    """

    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory
    )


# ---------------------------------------------------------
# 10. RECALL USER INFORMATION
# ---------------------------------------------------------

@tool
def recall_memory(query: str) -> str:
    """
    Recall long-term memories about the user that may answer their question. Call this before answering questions about personal facts or preferences shared earlier. Returns saved facts for the current chat thread.
    """

    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query=query
    )


# ---------------------------------------------------------
# 11. MAIN TOOL LIST
# ---------------------------------------------------------

tools = [
    # Existing tools
    openALex_search,
    search_uploaded_documents,
    scraper,
    remember_this,
    recall_memory,

    # New academic research tools
    scrape_selected_papers,
    ingest_academic_papers,
    academic_research_rag,
    academic_rag_statistics
]
