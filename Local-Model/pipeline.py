#  main components - indexing -> retreival -> generation
# for functionalities
import os
import pdfplumber
import time
import re
# to build retriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# to create vector store
from langchain_huggingface import HuggingFaceEmbeddings
# for the model
from llama_cpp import Llama
# to create the pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableMap, RunnableLambda

def load_and_clean_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    cleaned_text = re.sub(r"_+", "", full_text)
    return cleaned_text.strip()



def build_or_load_retriever() -> object:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists("faiss_index/index.faiss"):
        print("Loading existing FAISS retriever...")
        retriever = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True).as_retriever()

    else:
        print("Building new FAISS retriever...")
        # this is where you upload your document path
        docs = [Document(page_content=load_and_clean_pdf(r"<form-path-goes-here>"))]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
        vectorstore.save_local("faiss_index")
        retriever = vectorstore.as_retriever()
        print("FAISS retriever built and saved.")
        
    return retriever



# Format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt builder
def build_prompt(inputs) -> str:
    return f"""You are an intelligent assistant verifying life insurance proposal forms.

    Your job is to check if the form is:
    1. Structurally complete (all required sections present)
    2. Content-complete (no missing or invalid entries)

    Only use the context provided. Do not guess.

    If the form passes both checks, reply: "Document is valid and can be accepted."
    Else, explain clearly what is missing or incorrect.

    Context:
    {inputs['context']}

    Document:
    {inputs['document']}
    """

# model call
# specify the path of model here
def call_model(prompt):
    llm = Llama(model_path=r"<path-of-model-here>", n_ctx=8192)
    response = llm(prompt, temperature = 0.4, stop=["</s>"], max_tokens = 1024)
    return response["choices"][0]["text"]

# RAG chain
def call_rag(retriever):
  rag_chain = (
    RunnableMap({
        "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x['document']))),
        "document": RunnablePassthrough()
    })
    | RunnableLambda(build_prompt)
    | RunnableLambda(call_model)
  )

  return rag_chain

# Run it

def main():
  document = """Enter the form you want to test out"""
  
  print("Building retriever. . . ")
  retriever = build_or_load_retriever()
  rag_chain = call_rag(retriever)
  modelcall = time.time()
  response = rag_chain.invoke({'document':document})
  modelend = time.time()
  print(f'RAG:RESULT within {(modelend - modelcall)/60} minutes: {response}')


if __name__ == "__main__":
    main()
