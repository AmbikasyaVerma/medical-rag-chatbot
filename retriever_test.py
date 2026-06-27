from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 2})

docs = retriever.invoke("What are the symptoms of diabetes?")

print("\nRetrieved Chunks:\n")

for doc in docs:
    print(doc.page_content)
    print("-" * 50)