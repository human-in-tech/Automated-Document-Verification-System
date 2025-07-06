This manual describes how to run the **online version** of the Insurance Proposal Form Verifier, which uses Google Gemini API by default.

---

### 📂 Prerequisites

- Python 3.10+
- Internet connection
- Valid Gemini API key

---

### 🔧 Installation

1. **Clone the repository:**

```bash
git clone <repo-url>
cd project
```

2. **Create and activate virtual environment:**

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# or
source .venv/bin/activate  # On Unix/macOS
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Format the prompt and provide documents**
- Navigate to `pipeline.py`
- Specify the path of your policy documents in function `build_or_load_retriever()`. These will be guidelines based on which the LLM will judge the forms.
- Customise the prompt provided within `build_prompt()` to adjust according to your needs.

---

### 🔐 Set Up Gemini API

Set/Export your Gemini API key in environment via `pipeline.py`:

```bash
os.environ["GOOGLE_API_KEY"] = your-key-here
     # On Windows
```

---

### 🚀 Running the Online Pipeline


To test the model, craft the main function as per your requirements, and run:

```bash
python pipeline.py
```

To run the Streamlit UI:

```bash
streamlit run app.py
```

