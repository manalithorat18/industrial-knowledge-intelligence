# 🏭 Industrial Knowledge Intelligence

An AI-powered Retrieval-Augmented Generation (RAG) platform that enables users to upload multiple industrial documents such as manuals, SOPs, technical reports, and regulations, and interact with them using natural language.

The application combines **Google Gemini**, **LangChain**, **ChromaDB**, and **Streamlit** to provide document summarization, semantic search, conversational question answering, and persistent knowledge storage.

---

## 🚀 Features

- 📄 Upload multiple PDF documents simultaneously
- 🧠 Automatic text extraction and intelligent chunking
- 🔍 Semantic search using vector embeddings
- 🤖 AI-powered question answering using Google Gemini
- 💬 Conversational chat with context-aware responses
- 📊 Automatic document summarization
- 📚 Source chunk references for every answer
- 💾 Persistent ChromaDB knowledge base
- 📥 Download complete conversation history
- 📈 Dashboard displaying document statistics
- 🧹 Clear conversation functionality

---

# Architecture

```
                PDF Documents
                      │
                      ▼
              PDF Text Extraction
                  (PyPDF)
                      │
                      ▼
      Recursive Character Splitter
                      │
                      ▼
      HuggingFace Embeddings
                      │
                      ▼
             Chroma Vector Store
                      │
         Semantic Similarity Search
                      │
                      ▼
            Google Gemini 2.5 Flash
                      │
                      ▼
        Context-Aware Answer Generation
                      │
                      ▼
             Streamlit Web Interface
```

---

# Tech Stack

## Frontend

- Streamlit

## Backend

- Python

## AI / LLM

- Google Gemini 2.5 Flash

## Frameworks

- LangChain

## Vector Database

- ChromaDB

## Embedding Model

- sentence-transformers/all-MiniLM-L6-v2

## PDF Processing

- PyPDF

## Environment Management

- python-dotenv

---

# Project Structure

```
Industrial-Knowledge-Intelligence/
│
├── app.py
├── requirements.txt
├── .env
├── README.md
│
├── vectorstore/
│   └── chroma_db.py
│
├── db/
│   ├── chroma.sqlite3
│   └── ...
│
└── assets/
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/industrial-knowledge-intelligence.git

cd industrial-knowledge-intelligence
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file.

```
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

---

## Run Application

```bash
streamlit run app.py
```

---

# Workflow

### Step 1

Upload one or more PDF documents.

↓

### Step 2

The application extracts text from every PDF.

↓

### Step 3

Documents are split into overlapping chunks using LangChain.

↓

### Step 4

Each chunk is converted into vector embeddings using HuggingFace.

↓

### Step 5

Embeddings are stored in ChromaDB.

↓

### Step 6

The user asks a question.

↓

### Step 7

Relevant document chunks are retrieved using semantic similarity.

↓

### Step 8

Gemini receives:

- User Question
- Previous Conversation
- Retrieved Context

↓

### Step 9

Gemini generates a grounded response.

↓

### Step 10

The answer, source chunks, and chat history are displayed.

---

# Functionalities

## Document Upload

Supports uploading multiple PDF files simultaneously.

---

## Intelligent Chunking

Documents are split using:

- Chunk Size: 1000 characters
- Chunk Overlap: 200 characters

This preserves semantic context during retrieval.

---

## Semantic Search

Uses HuggingFace sentence embeddings and ChromaDB to retrieve the most relevant document chunks instead of performing keyword-based search.

---

## Conversational AI

Maintains previous chat history to support follow-up questions and context-aware responses.

---

## Document Summary

Generates a structured summary including:

- Executive Summary
- Key Equipment
- Procedures
- Risks
- Compliance
- Recommendations

---

## Source Attribution

Every answer includes the document chunks used for response generation along with relevance information.

---

## Conversation Export

Users can download the complete conversation as a text file.

---

# Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python |
| UI | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Framework | LangChain |
| Vector DB | ChromaDB |
| Embeddings | HuggingFace Sentence Transformers |
| PDF Reader | PyPDF |
| Environment | python-dotenv |

---

# Future Improvements

- Knowledge Graph Generation
- OCR Support for Scanned PDFs
- Multi-format document upload (DOCX, TXT, PPTX)
- User Authentication
- Cloud Deployment
- Document Versioning
- Hybrid Search (BM25 + Vector Search)
- Citation-aware Answer Generation
- Admin Dashboard
- Multi-user Knowledge Base

---

# Sample Questions

- What is an Operating System?
- Explain process scheduling.
- What are the responsibilities of a resource manager?
- Summarize the uploaded documents.
- What are the compliance requirements?
- Explain memory management.
- Compare scheduling algorithms.

---

# Author

**Manali Thorat**

B.Tech Computer Engineering

Pimpri Chinchwad College of Engineering

GitHub: https://github.com/manalithorat18

LinkedIn: https://linkedin.com/in/manali-thorat-423352292

---

# License

This project is intended for educational and research purposes.