# SpotMe RAG Chatbot

A RAG-based chatbot for SpotMe player data: SentenceTransformer embeddings + ChromaDB retrieval + LLM (Groq/Gemini) with function calling.

## Overview

This project implements a retrieval-augmented generation pipeline over SpotMe player data. Documents are chunked, embedded with SentenceTransformer, stored in ChromaDB, and retrieved via cosine similarity. The LLM (Groq or Gemini) then generates an answer grounded in the retrieved context, with function calling support. A simple RTL HTML frontend is included.

## Features

- Document chunking and embedding pipeline
- ChromaDB vector store
- Cosine-similarity retrieval
- LLM answer generation (Groq / Gemini) with function calling
- Session history in the chatbot orchestrator
- Simple HTML frontend

## Architecture

```text
Documents → Chunks → Embeddings (SentenceTransformer) → ChromaDB
                                                         ↓
User query → Embedding → Retrieval (cosine) → Context → LLM → Response
```

## Tech Stack

- Python
- FastAPI
- SentenceTransformers
- ChromaDB
- Groq / Gemini (LLM)

## Project Structure

```text
ChatBot_SBS_/
├── backend/
│   ├── embeddedmanger.py   # EmbeddingManager (SentenceTransformer)
│   ├── vectorstore.py      # VectorStore (ChromaDB wrapper)
│   ├── RAG.py              # RAGRetriever
│   ├── llm.py              # GroqLLM (tools)
│   └── spotmechatbot.py    # SpotMeChatbot (orchestrator + session history)
├── frontend/index.html
├── data/players.json
├── data_preparation.py     # players.json → chunks → embeddings → Chroma
├── main.py                 # FastAPI app (health / chat / reset)
├── requirements.txt
└── .env.example
```

## Installation

```bash
git clone https://github.com/ahmedyasser1588/ChatBot_SBS_.git
cd ChatBot_SBS_
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY
python data_preparation.py
python main.py          # or: uvicorn main:app --reload
```

## Usage

Open `http://localhost:8000` after starting the app.

## Project Status

Experimental — a RAG pipeline prototype.

## Future Improvements

- Add evaluation of retrieval quality
- Add tests
- Containerize with Docker
