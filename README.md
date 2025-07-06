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

### 1. Clone the Repository

```bash
git clone <repo-url>
cd project
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> 📝 For Offline GPU-optimized `llama-cpp-python` installation, use:
>
> ```bash
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/avx2
> ```

---

## Run the Application

### Online Version:

```bash
python pipeline.py  # uses Gemini API by default
```

### Offline Version:

Ensure GGUF model is downloaded and path is set in `pipeline.py`.

```bash
python pipeline.py  # uses local model on your system (mistral by default)
```

### Run Streamlit App (Both modes):

```bash
streamlit run app.py
```

---

## Prompt

> The assistant is guided by a structured prompt that checks for:
>
> 1. Presence of essential sections
> 2. Validity and completeness of content

The prompt has to be tweaked as per the requirements.

---


## Acknowledgements

* [LangChain](https://www.langchain.com/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Google Gemini](https://ai.google.dev/)
* [Llama.cpp](https://github.com/ggerganov/llama.cpp)

---

For setup help or model download instructions, checkout the manuals within local and online model directories.
