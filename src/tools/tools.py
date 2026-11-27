from langchain.tools import tool 
import requests
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from rich import print
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
import re 


load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))




import pymupdf

@tool
def extract_pdf_text(url):
    """Download a PDF and extract all its text."""

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    pdf = pymupdf.open(
        stream=response.content,
        filetype="pdf"
    )

    text = ""

    for page in pdf:
        text += page.get_text() + "\n"

    pdf.close()

    return text


def split_into_chunks(text, chunk_size=1500, overlap=150):
    """Split complete text into overlapping chunks."""

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append(text[start:end])

        start = end - overlap

    return chunks

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic.
    Returns titles, URLs and snippets."""

    pdf_query = f"{query} (filetype:pdf OR filetype:html)"

    MY_RESEARCH_DOMAINS = [
        "arxiv.org",
        "ncbi.nlm.nih.gov",
        "stanford.edu",
        "link.springer.com",
    ]

    results = tavily.search(
        query=pdf_query,
        max_results=3,
        include_domains=MY_RESEARCH_DOMAINS,
    )

    out = []

    for r in results.get("results", []):

        title = r.get("title") or "Title not available"
        url = r.get("url") or "URL not available"
        content = r.get("content") or "Snippet not available"

        out.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Snippet: {content[:300]}\n"
        )

    if not out:
        return "No relevant search results found."

    return "\n----\n".join(out)


   
    

@tool
def scrape_url(url: str) -> str:
    """
    Scrape and extract clean readable content from a URL.
    Uses multiple extraction strategies for better reliability.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    try:
        # ── Fetch page ─────────────────────────────────────
        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        html = response.text

        # ──────────────────────────────────────────────────
        # Strategy 1 → trafilatura (BEST for articles/blogs)
        # ──────────────────────────────────────────────────
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False
        )

        if extracted and len(extracted.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', extracted)
            return cleaned

        # ──────────────────────────────────────────────────
        # Strategy 2 → readability
        # ──────────────────────────────────────────────────
        doc = Document(html)
        clean_html = doc.summary()

        soup = BeautifulSoup(clean_html, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        if text and len(text.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', text)
            return cleaned

        # ──────────────────────────────────────────────────
        # Strategy 3 → fallback full page extraction
        # ──────────────────────────────────────────────────
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        cleaned = re.sub(r'\s+', ' ', text)

        if cleaned:
            return cleaned[:5000]

        return "Could not extract meaningful content from the page."

    except requests.exceptions.Timeout:
        return "Request timed out while scraping the URL."

    except requests.exceptions.HTTPError as e:
        return f"HTTP error occurred: {str(e)}"

    except Exception as e:
        return f"Could not scrape URL: {str(e)}"