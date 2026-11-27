
import os
import time
import requests
from datetime import date
from dotenv import load_dotenv

load_dotenv()

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")

OPENALEX_URL = "https://api.openalex.org/works"

# --------------------------------------------------
# 1. Dynamic publication-year range
# --------------------------------------------------

current_year = date.today().year
start_year = current_year - 10

# Semantic search supports publication_year,
# but not from_publication_date / to_publication_date.

semantic_filter = (
    f"publication_year:>{start_year - 1},"
    f"publication_year:<{current_year + 1},"
    "open_access.is_oa:true,"
    "type:article,"
    "is_retracted:false"
)

# Normal keyword search supports publication-date filters.

rerank_filter = (
    f"from_publication_date:{start_year}-01-01,"
    f"to_publication_date:{current_year}-12-31,"
    "open_access.is_oa:true,"
    "type:article,"
    "is_retracted:false"
)

# --------------------------------------------------
# 2. OpenAlex authentication
# --------------------------------------------------

headers = {}

if OPENALEX_API_KEY:
    headers["Authorization"] = f"Bearer {OPENALEX_API_KEY}"


# --------------------------------------------------
# 3. Get external PDF URLs
# --------------------------------------------------

def get_pdf_urls(paper):

    pdf_urls = []
    pdf_url_sources = {}

    locations = paper.get("locations") or []
    best_location = paper.get("best_oa_location")

    ordered_locations = []

    if best_location:
        ordered_locations.append(best_location)

    ordered_locations.extend(locations)

    for location in ordered_locations:

        if not location:
            continue

        source = location.get("source") or {}
        source_name = source.get("display_name", "Unknown source")

        pdf_url = location.get("pdf_url")

        if pdf_url and pdf_url not in pdf_urls:

            pdf_urls.append(pdf_url)

            pdf_url_sources[pdf_url] = source_name

    return pdf_urls, pdf_url_sources


# --------------------------------------------------
# 4. Main research paper search
# --------------------------------------------------

def search_research_papers(topic, limit=10):

    # -------------------------------
    # Semantic Search
    # -------------------------------

    semantic_params = {
        "search.semantic": topic,
        "filter": semantic_filter,
        "per_page": 50,
        "page": 1
    }

    response = requests.get(
        OPENALEX_URL,
        params=semantic_params,
        headers=headers,
        timeout=50
    )

    response.raise_for_status()

    semantic_results = response.json().get("results", [])

    time.sleep(1)

    # -------------------------------
    # Keyword Search with Reranking
    # -------------------------------

    rerank_params = {
        "search": topic,
        "rerank": "true",
        "filter": rerank_filter,
        "per_page": 100,
        "page": 1
    }

    response = requests.get(
        OPENALEX_URL,
        params=rerank_params,
        headers=headers,
        timeout=50
    )

    response.raise_for_status()

    rerank_results = response.json().get("results", [])

    # -------------------------------
    # Combine results by interleaving
    # -------------------------------

    combined_results = []

    max_length = max(
        len(semantic_results),
        len(rerank_results)
    )

    for i in range(max_length):

        if i < len(semantic_results):
            combined_results.append(semantic_results[i])

        if i < len(rerank_results):
            combined_results.append(rerank_results[i])

    # -------------------------------
    # Remove duplicate papers
    # -------------------------------

    unique_results = []
    seen_ids = set()

    for paper in combined_results:

        paper_id = paper.get("id")

        if paper_id and paper_id not in seen_ids:

            seen_ids.add(paper_id)
            unique_results.append(paper)

    # -------------------------------
    # Sort newest publications first
    # -------------------------------

    unique_results.sort(
        key=lambda paper: paper.get("publication_date") or "",
        reverse=True
    )

    # -------------------------------
    # Prepare paper metadata
    # -------------------------------

    papers = []

    for paper in unique_results:

        pdf_urls, pdf_url_sources = get_pdf_urls(paper)

        best_pdf_url = pdf_urls[0] if pdf_urls else None

        authors = []

        for authorship in paper.get("authorships", []):

            author = authorship.get("author", {})
            author_name = author.get("display_name")

            if author_name:
                authors.append(author_name)

        primary_location = paper.get("primary_location") or {}
        source = primary_location.get("source") or {}

        paper_data = {
            "openalex_id": paper.get("id"),
            "title": paper.get("title"),
            "year": paper.get("publication_year"),
            "publication_date": paper.get("publication_date"),
            "journal": source.get("display_name"),
            "journal_publisher": source.get("host_organization_name"),
            "volume": paper.get("biblio", {}).get("volume"),
            "cited_by_count": paper.get("cited_by_count"),
            "doi": paper.get("doi"),
            "authors": authors,
            "paper_type": paper.get("type"),
            "open_access": paper.get("open_access", {}).get("is_oa"),
            "pdf_url": best_pdf_url,
            "citation_pdf_url": best_pdf_url,
            "pdf_urls": pdf_urls,
            "pdf_url_sources": pdf_url_sources,
            "pdf_source": pdf_url_sources.get(best_pdf_url)
            if best_pdf_url else None
        }

        papers.append(paper_data)

    # -------------------------------
    # Return requested number of papers
    # -------------------------------

    return papers[:limit]
