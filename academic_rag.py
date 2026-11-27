
import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

from sentence_transformers import CrossEncoder


# ---------------------------------------------------------
# 2. ACADEMIC RAG CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ACADEMIC_DB_PATH = BASE_DIR / "academic_chroma_db"

ACADEMIC_DB_PATH.mkdir(parents=True, exist_ok=True)

ACADEMIC_COLLECTION_NAME = "academic_research_docs"


# ---------------------------------------------------------
# 3. EMBEDDING MODEL
# ---------------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# ---------------------------------------------------------
# 4. CHROMADB VECTOR STORE
# ---------------------------------------------------------

academic_vectorstore = Chroma(
    collection_name=ACADEMIC_COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=str(ACADEMIC_DB_PATH),
    collection_metadata={"hnsw:space": "cosine"}
)


# ---------------------------------------------------------
# 5. CROSS ENCODER
# ---------------------------------------------------------

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ---------------------------------------------------------
# 6. TEXT SPLITTER
# ---------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
)


# ---------------------------------------------------------
# 7. GENERATE UNIQUE PAPER ID
# ---------------------------------------------------------

def get_paper_id(paper: dict) -> str | None:

    # Accept both the old flat paper structure and
    # the new scraped-paper structure.

    metadata = paper.get("metadata") or paper

    openalex_id = metadata.get("openalex_id")

    if openalex_id:
        return f"openalex:{str(openalex_id).strip()}"

    doi = metadata.get("doi")

    if doi:
        doi = str(doi).strip().lower()

        doi = doi.replace("https://doi.org/", "")
        doi = doi.replace("http://doi.org/", "")
        doi = doi.replace("doi:", "")

        return f"doi:{doi}"

    pdf_url = (
        metadata.get("citation_pdf_url")
        or metadata.get("pdf_url")
        or metadata.get("extraction_pdf_url")
    )

    if pdf_url:
        return f"url:{str(pdf_url).strip()}"

    return None


# ---------------------------------------------------------
# 8. CHECK WHETHER PAPER ALREADY EXISTS
# ---------------------------------------------------------

def paper_already_exists(paper_id: str) -> bool:

    result = academic_vectorstore.get(
        where={"paper_id": paper_id},
        limit=1
    )

    return len(result["ids"]) > 0


# ---------------------------------------------------------
# 9. GENERATE UNIQUE CHUNK ID
# ---------------------------------------------------------

def create_chunk_id(paper_id: str, chunk_number: int) -> str:

    unique_value = f"{paper_id}_{chunk_number}"

    return hashlib.sha1(
        unique_value.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# 10. PREPARE PAPER METADATA
# ---------------------------------------------------------

def prepare_academic_metadata(
    paper: dict,
    paper_id: str,
    chunk_id: str
) -> dict:

    pdf_source = paper.get("pdf_source") or ""

    if isinstance(pdf_source, str):

        pdf_source_name = pdf_source
        pdf_source_publisher = ""
        pdf_source_type = ""

    elif isinstance(pdf_source, dict):

        pdf_source_name = pdf_source.get("name") or ""
        pdf_source_publisher = pdf_source.get("publisher") or ""
        pdf_source_type = pdf_source.get("type") or ""

    else:

        pdf_source_name = ""
        pdf_source_publisher = ""
        pdf_source_type = ""

    authors = paper.get("authors") or []

    if isinstance(authors, list):
        authors = ", ".join(
            str(author) for author in authors if author
        )

    citation_pdf_url = (
        paper.get("citation_pdf_url")
        or paper.get("pdf_url")
        or ""
    )

    metadata = {
        "paper_id": paper_id,
        "openalex_id": str(paper.get("openalex_id") or ""),
        "title": str(paper.get("title") or ""),
        "authors": str(authors),
        "year": int(paper.get("year") or 0),
        "publication_date": str(
            paper.get("publication_date") or ""
        ),
        "journal": str(paper.get("journal") or ""),
        "journal_publisher": str(
            paper.get("journal_publisher") or ""
        ),
        "volume": str(paper.get("volume") or ""),
        "doi": str(paper.get("doi") or ""),
        "cited_by_count": int(
            paper.get("cited_by_count") or 0
        ),
        "paper_type": str(
            paper.get("paper_type") or ""
        ),
        "open_access": bool(
            paper.get("open_access", False)
        ),

        # Original citation URL
        "pdf_url": str(citation_pdf_url),
        "citation_pdf_url": str(citation_pdf_url),

        # URL that actually provided the scraped content
        "extraction_pdf_url": str(
            paper.get("extraction_pdf_url") or ""
        ),

        # PDF source information
        "pdf_source_name": str(pdf_source_name),
        "pdf_source_publisher": str(pdf_source_publisher),
        "pdf_source_type": str(pdf_source_type),

        "chunk_id": chunk_id,
        "document_type": "academic"
    }

    return metadata


# ---------------------------------------------------------
# 11. ADD SCRAPED PAPERS TO CHROMADB
# ---------------------------------------------------------

def add_academic_papers_to_rag(
    papers: list[dict]
) -> dict:

    new_documents = []
    new_ids = []

    processed = 0
    skipped = 0
    failed = 0

    seen_paper_ids = set()

    for scraped_paper in papers:

        try:

            # ---------------------------------------------
            # A. GET SCRAPED CONTENT AND METADATA
            # ---------------------------------------------

            extracted_text = scraped_paper.get("content")

            paper = scraped_paper.get("metadata") or {}

            if not isinstance(extracted_text, str):
                print("Skipping paper: Content must be a string.")
                failed += 1
                continue

            if not extracted_text.strip():
                print("Skipping paper: Empty extracted content.")
                failed += 1
                continue

            if not isinstance(paper, dict):
                print("Skipping paper: Invalid metadata.")
                failed += 1
                continue

            # ---------------------------------------------
            # B. IDENTIFY PAPER
            # ---------------------------------------------

            paper_id = get_paper_id(paper)

            if not paper_id:
                print("Skipping paper: No unique identifier.")
                failed += 1
                continue

            # ---------------------------------------------
            # C. CHECK DUPLICATE PAPER
            # ---------------------------------------------

            if paper_id in seen_paper_ids:
                print(f"Duplicate in current batch: {paper_id}")
                skipped += 1
                continue

            seen_paper_ids.add(paper_id)

            if paper_already_exists(paper_id):
                print(f"Paper already exists: {paper_id}")
                skipped += 1
                continue

            print(f"\nProcessing scraped paper: {paper.get('title')}")

            # ---------------------------------------------
            # D. SPLIT SCRAPED TEXT INTO CHUNKS
            # ---------------------------------------------

            chunks = text_splitter.split_text(extracted_text)

            if not chunks:
                print(f"No chunks generated: {paper_id}")
                failed += 1
                continue

            # ---------------------------------------------
            # E. CREATE DOCUMENTS AND METADATA
            # ---------------------------------------------

            for chunk_number, chunk in enumerate(chunks):

                chunk_id = create_chunk_id(
                    paper_id,
                    chunk_number
                )

                metadata = prepare_academic_metadata(
                    paper,
                    paper_id,
                    chunk_id
                )

                document = Document(
                    page_content=chunk,
                    metadata=metadata
                )

                new_documents.append(document)
                new_ids.append(chunk_id)

            processed += 1

            print(
                f"Prepared paper: {paper.get('title')} "
                f"| Chunks: {len(chunks)}"
            )

        except Exception as e:

            print(
                f"Failed to process paper: "
                f"{scraped_paper.get('metadata', {}).get('title', 'Unknown')}: {e}"
            )

            failed += 1

    # -----------------------------------------------------
    # F. FINAL CHUNK-LEVEL DUPLICATE CHECK
    # -----------------------------------------------------

    documents_to_add = []
    ids_to_add = []
    if new_ids:

        existing_chunks = academic_vectorstore.get(
            ids=new_ids
        )

        existing_ids = set(existing_chunks["ids"])

        for document, chunk_id in zip(
            new_documents,
            new_ids
        ):

            if chunk_id not in existing_ids:
                documents_to_add.append(document)
                ids_to_add.append(chunk_id)

        # ---------------------------------------------
        # G. INSERT NEW CHUNKS
        # ---------------------------------------------

        if documents_to_add:

            academic_vectorstore.add_documents(
                documents=documents_to_add,
                ids=ids_to_add
            )

            print(
                f"\nInserted {len(documents_to_add)} "
                f"new academic chunks."
            )

    return {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "chunks_inserted": len(ids_to_add) if new_ids else 0
    }


# ---------------------------------------------------------
# 12. RECIPROCAL RANK FUSION
# ---------------------------------------------------------

def reciprocal_rank_fusion(
    rankings: list[list[Document]],
    rrf_k: int = 60
) -> list[Document]:

    rrf_scores = {}
    document_map = {}

    for ranking in rankings:

        for rank, document in enumerate(ranking, start=1):

            chunk_id = document.metadata.get("chunk_id")

            if not chunk_id:
                chunk_id = document.page_content

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

            document_map[chunk_id] = document

    sorted_ids = sorted(
        rrf_scores,
        key=lambda chunk_id: rrf_scores[chunk_id],
        reverse=True
    )

    return [
        document_map[chunk_id]
        for chunk_id in sorted_ids
    ]


# ---------------------------------------------------------
# 13. HYBRID ACADEMIC RETRIEVAL
# ---------------------------------------------------------

def retrieve_from_academic_rag(
    query: str,
    k: int = 15,
    candidate_k: int = 70
) -> str:

    print("\n" + "=" * 65)
    print("ACADEMIC HYBRID RAG RETRIEVAL STARTED")
    print("=" * 65)

    # -----------------------------------------------------
    # A. GET ALL STORED ACADEMIC DOCUMENTS
    # -----------------------------------------------------

    stored_data = academic_vectorstore.get(
        include=["documents", "metadatas"]
    )

    stored_texts = stored_data.get("documents") or []
    stored_metadatas = stored_data.get("metadatas") or []

    if not stored_texts:
        return "No academic documents are stored in the collection."

    all_documents = [
        Document(
            page_content=text,
            metadata=metadata or {}
        )
        for text, metadata in zip(
            stored_texts,
            stored_metadatas
        )
    ]

    print(f"Total stored chunks: {len(all_documents)}")

    # -----------------------------------------------------
    # B. ANN VECTOR RETRIEVAL
    # -----------------------------------------------------

    vector_results = (
        academic_vectorstore.similarity_search_with_score(
            query,
            k=min(candidate_k, len(all_documents))
        )
    )

    vector_documents = [
        document
        for document, distance in vector_results
    ]

    print(f"ANN retrieved: {len(vector_documents)}")

    # -----------------------------------------------------
    # C. BM25 KEYWORD RETRIEVAL
    # -----------------------------------------------------

    bm25_retriever = BM25Retriever.from_documents(
        all_documents,
        k=min(candidate_k, len(all_documents))
    )

    bm25_documents = bm25_retriever.invoke(query)

    print(f"BM25 retrieved: {len(bm25_documents)}")

    # -----------------------------------------------------
    # D. RECIPROCAL RANK FUSION
    # -----------------------------------------------------

    fused_documents = reciprocal_rank_fusion(
        [
            vector_documents,
            bm25_documents
        ]
    )

    print(f"After RRF: {len(fused_documents)}")

    # -----------------------------------------------------
    # E. CROSS ENCODER RERANKING
    # -----------------------------------------------------

    if not fused_documents:
        return "No relevant academic content found."

    pairs = [
        (query, document.page_content)
        for document in fused_documents
    ]

    rerank_scores = cross_encoder.predict(pairs)

    ranked_results = sorted(
        zip(rerank_scores, fused_documents),
        key=lambda item: item[0],
        reverse=True
    )

    final_documents = [
        document
        for score, document in ranked_results[:k]
    ]

    print(f"Final retrieved chunks: {len(final_documents)}")

    # -----------------------------------------------------
    # F. PREPARE CONTEXT WITH SOURCE METADATA
    # -----------------------------------------------------

    context_parts = []

    for index, document in enumerate(
        final_documents,
        start=1
    ):

        metadata = document.metadata

        context = f"""
ACADEMIC SOURCE {index}

Title: {metadata.get('title', 'Unknown')}
Authors: {metadata.get('authors', 'Unknown')}
Year: {metadata.get('year', 'Unknown')}
Journal: {metadata.get('journal', 'Unknown')}
DOI: {metadata.get('doi', 'Not available')}
Citations: {metadata.get('cited_by_count', 0)}
Citation PDF URL: {metadata.get('citation_pdf_url', metadata.get('pdf_url', 'Not available'))}
Extraction PDF URL: {metadata.get('extraction_pdf_url', 'Not available')}

Content:
{document.page_content}
"""

        context_parts.append(context)

    final_context = "\n\n".join(context_parts)

    print("=" * 65)
    print("ACADEMIC HYBRID RAG RETRIEVAL COMPLETED")
    print("=" * 65)

    return final_context


# ---------------------------------------------------------
# 14. COLLECTION STATISTICS
# ---------------------------------------------------------

def get_academic_rag_statistics() -> dict:

    stored_data = academic_vectorstore.get(
        include=["metadatas"]
    )

    metadatas = stored_data.get("metadatas") or []

    unique_papers = {
        metadata.get("paper_id")
        for metadata in metadatas
        if metadata and metadata.get("paper_id")
    }

    return {
        "total_chunks": len(metadatas),
        "total_unique_papers": len(unique_papers),
        "collection_name": ACADEMIC_COLLECTION_NAME,
        "storage_directory": str(ACADEMIC_DB_PATH)
    }
