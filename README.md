# 🧠 Chat with PDF

An AI-powered document assistant that allows users to **upload PDF files** and **interact conversationally** with their contents.  
The system leverages **Large Language Models (LLMs)**, **vector embeddings**, and **retrieval-based question answering** to deliver accurate and context-aware responses from any PDF document.

---

## 📖 Project Description

**Chat with PDF** is an intelligent application built using **Streamlit**, **LangChain**, and **OpenAI embeddings** that transforms static PDF files into interactive knowledge sources.

When a user uploads a PDF, the system:
1. Extracts and cleans the text content.  
2. Splits the text into smaller, manageable chunks for context handling.  
3. Converts these chunks into **vector embeddings** (numerical representations of text meaning).  
4. Stores the embeddings in a **vector database (FAISS)** for efficient semantic search.  
5. When a question is asked, the system retrieves the most relevant text segments and generates a **natural language answer** using a connected LLM (e.g., GPT model).

This enables users to query complex documents like research papers, reports, legal contracts, or manuals as if they were chatting with a knowledgeable assistant.  
It eliminates the need to manually search long PDFs and ensures answers are **context-specific, concise, and accurate**.

---

## ⚙️ Tech Stack
-------------------------------------------------------------------
| Category                   | Tools / Libraries                  |
|----------------------------|------------------------------------|
| **Language**               | Python 3.8+                        |
| **Framework**              | Streamlit                          |
| **AI / LLM**               | Gemini AI                          |
| **Vector Store**           | FAISS                              |
| **Document Processing**    | PyPDF2, LangChain Document Loaders |
| **Environment Management** | dotenv                             |
| **Frontend**               | Streamlit Web UI                   |
-------------------------------------------------------------------
---

## 🚀 Features

- 📄 Upload and process multiple PDF files  
- 💬 Chat interface to ask questions about document content  
- ⚡ Real-time, context-aware responses using embeddings + LLM  
- 🔍 Efficient retrieval of relevant information with FAISS  
- 🧩 Modular, easy-to-extend design 
- ☁️ Deployable on Streamlit Cloud or local environments  

---

## 🧰 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Nithyasri-12/Chat-with-PDF.git
cd Chat-with-PDF
