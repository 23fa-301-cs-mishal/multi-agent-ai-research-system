
from pathlib import Path
from typing import List
from collections import defaultdict
from dotenv import load_dotenv
import os
import certifi
import re
import math

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from pypdf import PdfReader
import docx2txt


Path("uploads").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)


# ==================================================
# 1. EMBEDDING MODEL
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# ==================================================
# 2. VECTOR STORE
# ==================================================

vectorstore = Chroma(
    collection_name="agentic_chatbot_docs",
    embedding_function=embeddings,
    persist_directory="chroma_db",
    collection_metadata={"hnsw:space": "cosine"}
)

# Verify the actual metric of the existing collection.
# Passing collection_metadata does not change an existing collection's metric.

collection_metadata = vectorstore._collection.metadata or {}

if collection_metadata.get("hnsw:space") != "cosine":
    raise ValueError(
        "The existing Chroma collection is not configured for cosine distance. "
        "Cosine scores cannot safely be calculated using 1 - distance. "
        "Check the existing collection configuration before continuing. "
        "The collection has NOT been deleted or recreated."
    )


# ==================================================
# 3. CROSS ENCODER MODEL
# ==================================================

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"
)


# ==================================================
# 4. READ UPLOADED FILES
# ==================================================

def read_file_text(file_path: str) -> str:

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":

        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"

        return text

    if suffix == ".docx":
        return docx2txt.process(file_path)

    if suffix in [".txt", ".md", ".py", ".csv"]:
        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    raise ValueError(
        "unsupported file type. upload pdf, docx, txt, md, py or csv"
    )


# ==================================================
# 5. ADD DOCUMENT TO RAG
# ==================================================

def add_document_to_rag(file_path: str, thread_id: str):

    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError("no text could be extracted from this file")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    docs: List[Document] = [

        Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "source": Path(file_path).name,
                "chunk_id": i + 1
            }
        )

        for i, chunk in enumerate(chunks)
    ]

    vectorstore.add_documents(docs)

    return {
        "filename": Path(file_path).name,
        "chunks": len(docs)
    }


# ==================================================
# 6. TOKENIZATION FOR BM25
# ==================================================

def tokenize(text: str) -> List[str]:

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ==================================================
# 7. SCORE STATISTICS
# ==================================================

def print_score_statistics(title, scores):

    print("\n" + "-" * 75)
    print(f"{title} STATISTICS")
    print("-" * 75)

    if not scores:
        print("No scores available.")
        return

    scores = [float(score) for score in scores]

    avg = sum(scores) / len(scores)

    variance = sum(
        (score - avg) ** 2
        for score in scores
    ) / len(scores)

    std_dev = math.sqrt(variance)

    sorted_scores = sorted(scores)
    middle = len(sorted_scores) // 2

    if len(sorted_scores) % 2 == 0:
        med = (
            sorted_scores[middle - 1] +
            sorted_scores[middle]
        ) / 2
    else:
        med = sorted_scores[middle]

    print(f"Count   : {len(scores)}")
    print(f"Maximum : {max(scores):.6f}")
    print(f"Minimum : {min(scores):.6f}")
    print(f"Mean    : {avg:.6f}")
    print(f"Median  : {med:.6f}")
    print(f"Std Dev : {std_dev:.6f}")



import math

def calculate_retrieval_rating(final_results):

    print("\n" + "=" * 75)
    print("NORMALIZED RETRIEVAL RELEVANCE RATING")
    print("=" * 75)

    if not final_results:
        print("No final results available.")
        return None

    # STEP 1: Extract final CrossEncoder scores
    scores = [
        score for doc, score in final_results
    ]

    total_chunks = len(scores)

    # STEP 2: Calculate average score
    average_score = sum(scores) / total_chunks

    print("\nSTEP 1: FINAL RETRIEVED CHUNKS")
    print(f"Total final chunks: {total_chunks}")
    print(f"CrossEncoder scores: {scores}")

    print("\nSTEP 2: CALCULATE AVERAGE SCORE")
    print("Formula: Mean = Sum of scores / Number of chunks")
    print(f"Mean = {sum(scores):.6f} / {total_chunks}")
    print(f"Mean = {average_score:.6f}")

    # STEP 3: Apply sigmoid normalization
    print("\nSTEP 3: APPLY SIGMOID NORMALIZATION")
    print("Formula: P = 1 / (1 + e^(-Mean))")

    normalized_score = 1 / (
        1 + math.exp(-average_score)
    )

    print(
        f"P = 1 / (1 + e^(-{average_score:.6f}))"
    )
    print(f"Normalized score = {normalized_score:.6f}")

    # STEP 4: Convert to rating out of 10
    print("\nSTEP 4: FINAL RATING")
    print("Formula: Rating = Normalized score * 10")

    rating = normalized_score * 10

    print(
        f"Rating = {normalized_score:.6f} * 10"
    )
    print(f"Final Rating = {rating:.2f}/10")

    print("\n" + "=" * 75)

    return rating



# ==================================================
# 8. BM25 KEYWORD RETRIEVAL
# ==================================================

def bm25_search(
    query: str,
    documents: List[Document],
    k: int
):

    if not documents:
        return []

    tokenized_documents = [
        tokenize(doc.page_content)
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices:

        if scores[index] <= 0:
            continue

        results.append(
            (documents[index], float(scores[index]))
        )

        if len(results) >= k:
            break

    return results


# ==================================================
# 9. RECIPROCAL RANK FUSION (RRF)
# ==================================================

def reciprocal_rank_fusion(
    ann_results,
    bm25_results,
    rrf_k: int = 60
):

    fused_scores = {}
    documents = {}

    retrieval_lists = [
        [doc for doc, score in ann_results],
        [doc for doc, score in bm25_results]
    ]

    for result_list in retrieval_lists:

        for rank, doc in enumerate(result_list, start=1):

            key = (
                doc.metadata.get("source", ""),
                doc.page_content
            )

            documents[key] = doc

            fused_scores[key] = (
                fused_scores.get(key, 0)
                + 1 / (rrf_k + rank)
            )

    ranked_keys = sorted(
        fused_scores,
        key=lambda key: fused_scores[key],
        reverse=True
    )

    return [
        (documents[key], fused_scores[key])
        for key in ranked_keys
    ]


# ==================================================
# 10. CROSS ENCODER RERANKING
# ==================================================

def rerank_documents(
    query: str,
    documents
):

    if not documents:
        return []

    pairs = [
        (query, doc.page_content)
        for doc, score in documents
    ]

    scores = cross_encoder.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )

    ranked_documents = sorted(
        zip(
            [doc for doc, score in documents],
            scores
        ),
        key=lambda item: float(item[1]),
        reverse=True
    )

    return [
        (doc, float(score))
        for doc, score in ranked_documents
    ]


# ==================================================
# 11. SELECT FINAL CHUNKS
# ==================================================

def select_final_chunks(reranked_results, k):

    return reranked_results[:k]


# ==================================================
# 12. ASSIGN CHUNK NUMBERS
# ==================================================

def assign_chunk_numbers(documents):

    source_counters = defaultdict(int)
    chunk_number_map = {}

    for doc in documents:

        source = doc.metadata.get("source", "")

        # Preserve original chunk IDs when available.
        # Older stored documents receive display numbers.

        if "chunk_id" not in doc.metadata:

            source_counters[source] += 1

            doc.metadata["chunk_id"] = source_counters[source]

        key = (
            source,
            doc.page_content
        )

        chunk_number_map.setdefault(
            key,
            doc.metadata["chunk_id"]
        )

    return chunk_number_map


def apply_chunk_numbers(documents, chunk_number_map):

    for doc in documents:

        key = (
            doc.metadata.get("source", ""),
            doc.page_content
        )

        doc.metadata["chunk_id"] = chunk_number_map.get(
            key,
            doc.metadata.get("chunk_id", "Unknown")
        )


# ==================================================
# 13. PRINT RANKED RESULTS
# ==================================================

def print_ranked_results(title, results, score_name):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    if not results:
        print("No results found.")
        return

    for rank, (doc, score) in enumerate(results, start=1):

        chunk_id = doc.metadata.get("chunk_id", "Unknown")
        source = doc.metadata.get("source", "Unknown")

        print(
            f"Rank: {rank} | "
            f"Source: {source} | "
            f"Chunk Number: {chunk_id} | "
            f"{score_name}: {score:.6f}"
        )

    print(f"\nTotal results: {len(results)}")

    print_score_statistics(
        title,
        [score for doc, score in results]
    )


# ==================================================
# 14. CROSS ENCODER RESULTS AND STATISTICS
# ==================================================

def print_cross_encoder_results(
    reranked_results,
    final_k
):

    print("\n" + "=" * 75)
    print("CROSS ENCODER RESULTS")
    print("=" * 75)

    if not reranked_results:
        print("No results found.")
        return

    scores = [
        score
        for doc, score in reranked_results
    ]

    print_score_statistics(
        "CROSS ENCODER",
        scores
    )

    print(f"\nFinal K: {final_k}")

    print("\nCOMPLETE RERANKED CHUNKS")

    for rank, (doc, score) in enumerate(
        reranked_results,
        start=1
    ):

        chunk_id = doc.metadata.get(
            "chunk_id",
            "Unknown"
        )

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        print("\n" + "=" * 75)

        if rank <= final_k:
            print("<FINAL SELECTED>")
        else:
            print("<NOT IN FINAL SELECTION>")

        print(f"Rank: {rank}")
        print(f"Chunk Number: {chunk_id}")
        print(f"Source: {source}")
        print(f"CrossEncoder Score: {score:.6f}")

        print("\nFull Chunk Content:")
        print("-" * 75)
        print(doc.page_content)
        print("-" * 75)

    print("\n" + "=" * 75)


# ==================================================
# 15. FINAL RAG RETRIEVAL PIPELINE
# ==================================================

def retrieve_from_rag(
    query: str,
    thread_id: str,
    k: int = 15,
    candidate_k: int = 70
) -> str:

    print("\n" + "#" * 75)
    print("HYBRID RAG RETRIEVAL STARTED")
    print("#" * 75)

    print(f"Query: {query}")
    print(f"Thread ID: {thread_id}")
    print(f"Final K: {k}")

    # --------------------------------------------------
    # STEP 1: GET DOCUMENTS FOR THIS THREAD
    # --------------------------------------------------

    stored_data = vectorstore.get(
        where={"thread_id": thread_id},
        include=["documents", "metadatas"]
    )

    stored_documents = stored_data.get("documents") or []
    stored_metadatas = stored_data.get("metadatas") or []

    if not stored_documents:
        print("\nNo uploaded documents found.")
        return "no relevant uploaded document content found"

    documents = [
        Document(
            page_content=content,
            metadata=metadata or {}
        )
        for content, metadata in zip(
            stored_documents,
            stored_metadatas
        )
    ]

    chunk_number_map = assign_chunk_numbers(documents)

    candidate_count = min(
        candidate_k,
        len(documents)
    )

    print(f"\nTotal stored chunks: {len(documents)}")
    print(f"Candidates per retriever: {candidate_count}")

    # --------------------------------------------------
    # STEP 2: ANN SEARCH
    # --------------------------------------------------

    print("\n[STEP 1] ANN SEARCH")

    ann_raw_results = vectorstore.similarity_search_with_score(
        query=query,
        k=candidate_count,
        filter={"thread_id": thread_id}
    )

    # Chroma cosine distance = 1 - cosine similarity

    ann_results = [
        (doc, 1 - float(distance))
        for doc, distance in ann_raw_results
    ]

    ann_results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    apply_chunk_numbers(
        [doc for doc, score in ann_results],
        chunk_number_map
    )

    print_ranked_results(
        "ANN RESULTS",
        ann_results,
        "Cosine Similarity"
    )

    # --------------------------------------------------
    # STEP 3: BM25 SEARCH
    # --------------------------------------------------

    print("\n[STEP 2] BM25 SEARCH")

    bm25_results = bm25_search(
        query=query,
        documents=documents,
        k=candidate_count
    )

    print_ranked_results(
        "BM25 RESULTS",
        bm25_results,
        "BM25 Score"
    )

    # --------------------------------------------------
    # STEP 4: RRF
    # --------------------------------------------------

    print("\n[STEP 3] RECIPROCAL RANK FUSION")

    fused_results = reciprocal_rank_fusion(
        ann_results=ann_results,
        bm25_results=bm25_results,
        rrf_k=60
    )

    print_ranked_results(
        "RRF RESULTS",
        fused_results,
        "RRF Score"
    )

    # --------------------------------------------------
    # STEP 5: CROSS ENCODER
    # --------------------------------------------------

    print("\n[STEP 4] CROSS ENCODER RERANKING")

    reranked_results = rerank_documents(
        query=query,
        documents=fused_results
    )

    print_cross_encoder_results(
        reranked_results,
        k
    )

    # --------------------------------------------------
    # STEP 6: FINAL SELECTION
    # --------------------------------------------------

    final_results = select_final_chunks(
        reranked_results,
        k
    )

    if not final_results:
        print("\nNo relevant chunks found.")
        return "no relevant uploaded document content found"
    # Calculate overall retrieval rating
    calculate_retrieval_rating(final_results)

    final_scores = [
        score
        for doc, score in final_results
    ]

    average_score = sum(final_scores) / len(final_scores)

    print("\n" + "=" * 75)
    print("FINAL RETRIEVAL SUMMARY")
    print("=" * 75)

    print(f"Total stored chunks: {len(documents)}")
    print(f"ANN retrieved: {len(ann_results)}")
    print(f"BM25 retrieved: {len(bm25_results)}")
    print(f"Unique chunks after RRF: {len(fused_results)}")
    print(f"CrossEncoder reranked: {len(reranked_results)}")
    print(f"Final chunks selected: {len(final_results)}")
    print(f"Average final CrossEncoder score: {average_score:.6f}")

    print("\nHYBRID RAG RETRIEVAL COMPLETED")

    # --------------------------------------------------
    # STEP 7: FORMAT FINAL CHUNKS FOR THE AGENT
    # --------------------------------------------------

    results = []

    for i, (doc, score) in enumerate(
        final_results,
        start=1
    ):

        source = doc.metadata.get(
            "source",
            "uploaded document"
        )

        chunk_id = doc.metadata.get(
            "chunk_id",
            "Unknown"
        )

        results.append(
            f"[Source: {source} | Chunk: {chunk_id}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(results)


# ==================================================
# 16. TEST RAG USING A SINGLE DOCUMENT
# ==================================================

def test_single_document():

    file_path = r"C:\Users\CBM\Desktop\scraped_arxiv_2411.11901.txt"

    test_thread_id = "single_document_test"

    print("\n" + "#" * 75)
    print("SINGLE DOCUMENT RAG TEST")
    print("#" * 75)

    if not Path(file_path).exists():
        print("\nERROR: Document not found!")
        print(file_path)
        return

    print(f"\nDocument: {Path(file_path).name}")

    # Check whether this document is already stored.
    # Do not delete existing data or recreate the collection.

    existing_data = vectorstore.get(
        where={"thread_id": test_thread_id},
        include=["documents"]
    )

    existing_documents = existing_data.get("documents") or []

    if existing_documents:

        print("\nExisting document chunks found.")
        print("Using existing Chroma data. No re-ingestion needed.")

    else:

        print("\nNo existing test chunks found.")

        upload_result = add_document_to_rag(
            file_path=file_path,
            thread_id=test_thread_id
        )

        print("\nDocument successfully added!")
        print(f"Filename: {upload_result['filename']}")
        print(f"Total chunks: {upload_result['chunks']}")

    query = input("\nEnter your research query: ").strip()

    if not query:
        print("\nERROR: Query cannot be empty.")
        return

    final_context = retrieve_from_rag(
        query=query,
        thread_id=test_thread_id,
        k=15,
        candidate_k=70
    )

    print("\nFinal context prepared for agent.")
    print(f"Context characters: {len(final_context)}")

    print("\nTEST COMPLETED")


# ==================================================
# 17. RUN TEST
# ==================================================

if __name__ == "__main__":
    test_single_document()
