# ✨ RAG Knowledge Assistant

> An AI-powered multi-modal assistant built with LangChain, Google Gemini, and Streamlit.  
> Upload documents, images, or data files — ask anything — get grounded answers powered by Gemini 1.5 Flash.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2.15-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-red)
![Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-orange)

---

## 🌐 Live Demo

> **[▶ Try it here →](https://rag-assistant-qfhteykrenpwycenjkaxpa.streamlit.app)**

---

## 📸 Screenshots

![Main Interface](assets/screenshot_main1.png)
![Main Interface](assets/screenshot_main2.png)
![Q&A Result](assets/screenshot_qa1.png)
![Quiz Feature](assets/screenshot_quiz1.png)

---

## 🚀 Features

- **Document Q&A** — Upload PDF, TXT, DOCX and ask questions grounded in your content, with source attribution
- **RAG Pipeline** — Semantic chunking → Google Embeddings → Chroma Vector DB → Retrieval → Gemini generation
- **Score Goal Mode** — Adjusts answer depth by target grade: Pass / Credit / Distinction / High Distinction
- **Quiz Generator** — Auto-generates 3 practice questions from document content
- **Image Analysis** — Multi-modal visual Q&A via Gemini 1.5 Flash (e-commerce, contract, competitor, interior, medical modes)
- **Data Analysis** — Natural language queries on CSV/Excel + AI-generated downloadable charts
- **Multi-Session** — Independent sessions with custom names, pin to top, and color labels

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Streamlit Frontend            │
│  Multi-Session │ Chat UI │ Quiz Panel   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│              RAG Pipeline               │
│  Load → Chunk → Embed → Store → Retrieve│
└──────┬─────────────────┬────────────────┘
       │                 │
┌──────▼──────┐  ┌───────▼────────────────┐
│   Chroma    │  │   Google Gemini API    │
│ Vector DB   │  │  gemini-1.5-flash (LLM)│
│  (local)    │  │  embedding-001 (embed) │
└─────────────┘  └────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.56.0 |
| LLM | Google Gemini 1.5 Flash |
| Vision | Google Gemini 1.5 Flash (multi-modal) |
| Embedding | Google text-embedding-001 |
| Vector DB | Chroma (local) |
| RAG Framework | LangChain + langchain-chroma |
| Document Loaders | PyMuPDF · Docx2txt · TextLoader |
| Data Analysis | Pandas · Matplotlib |

---

## ⚙️ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/xxz212/rag-assistant.git
cd rag-assistant
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Open .env and fill in your Google Gemini API key
```

Get your free key at: https://aistudio.google.com  
*(No credit card required)*

### 3. Run

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
rag-assistant/
├── app.py              # Streamlit UI + session management
├── rag.py              # RAG pipeline (load/chunk/embed/retrieve/answer)
├── vision.py           # Image analysis (Gemini vision)
├── data_analysis.py    # CSV/Excel analysis + chart generation
├── session.py          # Session state management
├── requirements.txt    # Pinned dependencies
├── .env.example        # Environment variable template
└── assets/             # Screenshots
```

---

## 💡 Key Design Decisions

**Why Google Gemini 1.5 Flash?**  
Free tier with generous quota — suitable for portfolio demos without billing. Multi-modal support (text + vision) in a single model eliminates the need for separate vision APIs.

**Why RAG instead of fine-tuning?**  
RAG keeps answers grounded in uploaded documents, preventing hallucination. No training cost, instantly updates when new documents are uploaded.

**Why session-based architecture?**  
Each conversation maintains independent vector stores and history — enables multi-document workflows without cross-session data leakage.

---

## 🗺️ Roadmap

- [x] Multi-format document Q&A (PDF / TXT / DOCX)
- [x] Image analysis with 5 domain modes
- [x] Data analysis + chart generation
- [x] Score-based answer depth adjustment
- [x] Streamlit Cloud deployment
- [ ] Streaming output (token-by-token)
- [ ] Persistent vector store (survive page refresh)
- [ ] REST API (FastAPI backend)

---

## 👨‍💻 Author

Built by **Nate** — Master of IT @ University of Wollongong  
Targeting AI Application Engineer roles in Melbourne / Sydney.

[![GitHub](https://img.shields.io/badge/GitHub-xxz212-black)](https://github.com/xxz212)
