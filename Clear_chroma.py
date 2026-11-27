import chromadb

client = chromadb.PersistentClient(
    path="chroma_db"
)

client.delete_collection(
    name="agentic_chatbot_docs"
)

print("Old collection deleted successfully!")

print(
    "Remaining collections:",
    client.list_collections()
)