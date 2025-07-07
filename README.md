# Automated Document Verification System

This project is a document verification assistant designed to validate forms for structure and content completeness. It supports both **online** (using model APIs) and **offline** (using local GGUF LLMs) for verification pipelines.

---

## Features

* Upload PDF forms for validation.
* Retrieves relevant chunks from policy documents using FAISS + sentence-transformer embeddings.
* Uses a custom prompt + LLM to verify completeness and structure.
* Supports both API-based and local (GGUF LLMs) inference.
* Fast response and privacy-friendly offline deployment.

---

## Tech Stack

* **LLM API (Online)**: Google Gemini Pro / Flash
* **LLM (Offline)**: Mistral 7B / TinyLLaMA in GGUF format via `llama-cpp-python`
* **RAG Pipeline**: LangChain core + HuggingFaceEmbeddings + FAISS
* **Frontend**: Streamlit

---

## Setup Instructions

For setup help or model download instructions, checkout the manuals within local and online model directories.


---

