# Medical RAG Chatbot

A focused Retrieval Augmented Generation (RAG) chatbot for answering questions from medical PDF documents. The project includes a Streamlit interface for uploading a PDF and a small command-line workflow for indexing and querying a local document collection.

> Educational use only: this application is not a medical device and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Project Overview

Medical RAG Chatbot demonstrates the core RAG pattern in a clean, portfolio-ready form: load a medical PDF, split it into chunks, embed the chunks, retrieve the most relevant context for a user question, and ask a Groq-hosted language model to answer from that context.

## Features

- Streamlit web app for PDF upload and document-grounded chat.
- Local PDF ingestion script for building a reusable Chroma vector database.
- Hugging Face sentence-transformer embeddings.
- Groq LLM integration through LangChain.
- Strict prompt behavior that asks the model to answer from retrieved context.
- Safe environment variable workflow using `.env.example`.
- Clean repository layout for independent GitHub publishing.

## Architecture

1. `app.py` accepts a PDF upload and stores it temporarily for processing.
2. `PyPDFLoader` extracts document text from the PDF.
3. `RecursiveCharacterTextSplitter` creates overlapping chunks for retrieval.
4. `HuggingFaceEmbeddings` converts chunks into vectors.
5. ChromaDB stores vectors and retrieves the most relevant chunks for a question.
6. `ChatGroq` receives the retrieved context and generates a concise answer.
7. `rag_chat.py` provides the same RAG idea through a command-line interface using a persisted local Chroma database.

## Technologies Used

- Python
- Streamlit
- LangChain Community
- LangChain Text Splitters
- LangChain Groq
- ChromaDB
- Sentence Transformers
- PyPDF
- python-dotenv

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variable Setup

Create a local `.env` file from the provided template:

```bash
copy .env.example .env
```

Add your Groq API key to `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The `.env` file is ignored by Git and should never be committed.

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

For the command-line workflow, place a local PDF at `data/cancer.pdf`, build the vector database, then ask questions:

```bash
python ingest.py
python rag_chat.py
```

The generated `chroma_db/` directory is local runtime data and is intentionally ignored by Git.

## Example Queries

- What are the main symptoms discussed in the document?
- What risk factors does the PDF mention?
- Summarize the prevention guidance.
- What treatment options are described?
- Does the document mention when to consult a doctor?

## Folder Structure

```text
medical-rag-chatbot/
  app.py              # Streamlit PDF upload and chat interface
  ingest.py           # Builds a local Chroma vector database from data/cancer.pdf
  rag_chat.py         # Command-line RAG chatbot
  retriever_test.py   # Retrieval sanity check
  requirements.txt    # Minimal runtime dependencies
  .env.example        # Safe environment variable template
  .gitignore          # Excludes secrets, caches, virtualenvs, and local databases
  data/
    .gitkeep          # Keeps the data folder without committing private PDFs
```

## Future Improvements

- Add page-number citations for retrieved evidence.
- Support multiple uploaded PDFs in a single session.
- Add automated tests for ingestion and retrieval behavior.
- Add document metadata filtering.
- Add deployment instructions for Streamlit Community Cloud.
- Improve UI state handling for repeated questions over the same document.