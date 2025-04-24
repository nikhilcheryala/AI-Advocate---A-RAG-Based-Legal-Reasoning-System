
# ⚖️ AI Advocate –A RAG Based Legal Document Analysis System

AI Advocate is a Retrieval-Augmented Generation (RAG)-based system designed to analyze, summarize, and answer legal queries from uploaded PDF documents. Built with LangChain, HuggingFace Transformers, FAISS, and Streamlit, this full-stack AI application assists users in understanding complex legal documents.

---

## 🚀 Features

- 📂 Upload and parse legal PDF documents
- 🔍 Chunk and index content using FAISS vector database
- 🧠 Semantic search powered by HuggingFace sentence transformers
- 🤖 Legal reasoning and question answering with Groq-hosted DeepSeek LLM
- 📋 Document summarization with structured output
- 📎 Citation tracking linked to specific pages in the PDF
- 📄 Downloadable report generator with query-answer history
- 🌐 Intuitive Streamlit interface with dark mode UI

---

## 🛠️ Technology Stack

| Layer       | Tools & Libraries |
|-------------|-------------------|
| Frontend    | Streamlit         |
| Backend     | Python, LangChain |
| Embeddings  | HuggingFace (`all-mpnet-base-v2`) |
| Vector Store| FAISS             |
| LLM         | DeepSeek via Groq |
| PDF Parsing | PDFPlumber        |
| Report Gen  | ReportLab         |

---

## 📦 Installation

```bash
git clone https://github.com/nikhilcheryala/AI-Advocate---A-RAG-Based-Legal-Reasoning-System.git
cd ai-advocate
pip install -r requirements.txt
```

---

## 🔑 Setup Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📁 Folder Structure

```
ai-advocate/
├── frontend.py                 # Streamlit app
├── rag_pipeline.py            # Core RAG logic (LLM + prompt + summarizer)
├── vector_database.py         # PDF parsing, embedding, retrieval
├── pdfs/                      # Uploaded PDFs
├── vectorstore/               # FAISS index storage
├── requirements.txt           # All dependencies
└── .env                       # API keys
```

---

## ▶️ Running the App

```bash
streamlit run frontend.py
```

---

## ✅ How to Use

1. Upload a legal PDF.
2. Click **Summarize Document** to view a structured summary.
3. Ask questions about the document in natural language.
4. Download a full PDF report of the Q&A session.

---

## 📚 Citation Tracking

The AI provides source references in summaries like:
> "This clause allows remarriage after divorce. *(Source: Page 3)*"

This improves transparency and traceability in legal analysis.

---

## 📈 Future Scope

- Multilingual support for regional legal documents
- Real-time citation mapping to highlighted PDF content
- Cloud deployment and secure login

---

## 🤝 Contributions

PRs and feedback are welcome. Please fork the repository and submit a pull request.

---

## 📄 License

MIT License © 2025 Cheryala Nikhil
