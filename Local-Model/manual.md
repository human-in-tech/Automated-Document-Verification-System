This manual describes how to set up and run the **offline version** of the Insurance Proposal Form Verifier, using a local LLM (Mistral 7B by default).

---

### 📂 Prerequisites

- Python 3.10
- At least **16 GB RAM** for Mistral
- No GPU required
- Compatible with Windows (Hasn't been tested on other OS yet)

---

### 🔧 Installation & Setup

1. **Clone the repository:**

```bash
git clone https://github.com/human-in-tech/Automated-Document-Verification-System
cd Automated-Document-Verification-System/Local-Model
```

2. **Create and activate virtual environment:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> If `llama-cpp-python` fails, install using:
>
> ```bash
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/avx2
> ```
> or if you have conda installed, use:
> ```bash
> conda install -c conda-forge llama-cpp-python
> ```

4. **Download GGUF model**

- Choose a model from HuggingFace or other sources (e.g., `mistral-7b-instruct-v0.2.Q4_K_M.gguf`)
- Place it in the `models/` directory

5. **Customise the prompt and policy documents**
- Navigate to `pipeline.py`.
- Specify the path of your policy documents in function `build_or_load_retriever()`. These will be guidelines based on which the LLM will judge the forms.
- Customise the prompt provided within `build_prompt()` to adjust according to your needs.

---

### 🚀 Running the Offline Pipeline

In `pipeline.py`, make sure the `model_path` give within the function `call_model()` points to your downloaded GGUF model:

```python
llm = Llama(model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=8192)
```

To test the model itself, run:

```bash
python pipeline.py
```

To launch the user interface, run :

```bash
streamlit run app.py
```
