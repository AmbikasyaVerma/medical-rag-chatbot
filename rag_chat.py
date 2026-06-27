from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load ChromaDB
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 2})

# LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

# User Question
question = input("Ask a question: ")

# Retrieve Relevant Chunks
docs = retriever.invoke(question)

print("\nRETRIEVED CHUNKS:\n")

for doc in docs:
    print(doc.page_content)
    print("\n" + "=" * 50 + "\n")

# Build Context
context = "\n\n".join(
    [doc.page_content for doc in docs]
)

# Prompt
prompt = f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I could not find that information in the PDF."
"""

# Generate Answer
response = llm.invoke(prompt)

print("\nANSWER:\n")
print(response.content)