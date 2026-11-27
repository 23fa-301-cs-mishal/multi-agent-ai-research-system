from Search import search_research_papers
from academic_rag import (
    add_academic_papers_to_rag,
    retrieve_from_academic_rag,
    get_academic_rag_statistics
)

# 1. OPENALEX SEARCH
query = input("Enter your query: ")

results = search_research_papers(query, limit=10)

# 2. DISPLAY PAPER INFORMATION
for index, paper in enumerate(results, start=1):
    print(f"\nPAPER: {index}")
    print("Title:", paper["title"])
    print("Year:", paper["year"])
    print("PDF URL:", paper["pdf_url"])

# 3. INSERT PAPERS INTO ACADEMIC RAG
result = add_academic_papers_to_rag(results)

print("\nINGESTION RESULT:")
print(result)

# 4. DISPLAY COLLECTION STATISTICS
print("\nACADEMIC RAG STATISTICS:")
print(get_academic_rag_statistics())

# 5. TEST ACADEMIC RETRIEVAL
question = input("\nEnter your research question: ")

context = retrieve_from_academic_rag(
    query=question,
    k=15,
    candidate_k=70
)

print("\nRETRIEVED ACADEMIC CONTENT:")
print(context)