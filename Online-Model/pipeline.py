#  main components - indexing -> retreival -> generation

# !pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain sentence-transformers bs4 -q

# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import google.generativeai as genai
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain.schema import Document
import pdfplumber
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


os.environ["GOOGLE_API_KEY"] = 'AIzaSyB722ymFP1U4wZGHR98O-8BE8moHtVOyvI'
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
llm = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-pro"

  
def load_and_clean_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    # Clean text: remove long underscores and excessive empty lines
    cleaned_text = re.sub(r"_+", "", full_text)  # Remove ____ lines
    return cleaned_text.strip()

def build_or_load_retriever() -> object:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists("faiss_index/index.faiss"):
        print("Loading existing FAISS retriever...")
        retriever = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True).as_retriever()

    else:
        print("Building new FAISS retriever...")
        docs = [Document(page_content=load_and_clean_pdf(r"C:\Users\khyat\OneDrive\Desktop\DRDO\Project\proposal forms\icici proposal form.pdf"))]
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
    return f"""You are an intelligent document verification assistant for a life term insurance company. 
    Your job is to assess whether a submitted proposal form is valid or not.
    Only use the provided context to make your judgment. Do not guess or assume missing information.
    Evaluate the document in two phases:
        1. Format and Structure Verification
        2. Content Completeness Verification

    If the document passes both phases, respond: "Document is valid and can be accepted."
    Otherwise, provide a clear explanation of what is missing or incorrect.
    context: {inputs['context']}

    document to be verified: {inputs['document']}
    """

# Gemini call
def call_gemini(prompt):
    return llm.generate_content(prompt).text

# RAG chain
def call_rag(retriever):
  rag_chain = (
    RunnableMap({
        "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x['document']))),
        "document": RunnablePassthrough()
    })
    | RunnableLambda(build_prompt)
    | RunnableLambda(call_gemini)
  )

  return rag_chain

# Run it

def main():
  document = """{
  "GenericDetails": {
    "PolicyNo": "IC112233",
    "ClientID": "CL556677",
    "IsPolicySelfProposed": "Yes",
    "TypeOfProposer": "Individual",
    "RelationshipWithLifeAssured": "Spouse",
    "TypeOfProposal": "Individual"
  },
  "ProposerPolicyOwnerDetails": {
    "CommunicationAddress": {
      "LINE1": "221B Baker Street",
      "LINE2": "",
      "LANDMARK": "Next to old library",
      "CITY": "Delhi",
      "STATE": "Delhi",
      "COUNTRY": "India",
      "PinCode": "110001"
    },
    "EmailID": "johndoegmail.com",  
    "DOB": "1991-13-01",
    "Gender": "Male",
    "Nationality": "Indian",
    "MaritalStatus": "Married",
    "Education": "Diploma",
    "Occupation": "Salaried",
    "Mobile": "987654321",
    "CountryCode": "+91",
    "ReceiveEmail": true,
    "ReceiveSMS": true,
    "PermanentAddress": {
      "LINE1": "Flat 303, Palm Residency",
      "LINE2": "Near Bus Depot",
      "LANDMARK": "",
      "CITY": "Delhi",
      "STATE": "Delhi",
      "COUNTRY": "India",
      "PinCode": "110001"
    },
    "IndustryType": "Healthcare",
    "OrganisationType": "Pvt. Ltd.",
    "IncomeAnnual": "120000", 
    "OrganisationName": "MediCare Pvt Ltd",
    "SharePortfolioWithAgent": "Yes",
    "PoliticallyExposedPerson": "No",
    "AddressProof": "Voter ID",
    "IncomeProof": "Bank Statement",
    "IdentityProof": "PAN Card",
    "IdentityProofNumber": "ABCDE1234", 
    "IdentityProofExpiry": "",
    "PAN": "ABCDE1234", 
    "OtherDocument": "",
    "ExistingKYCNumber": "1234567890123456",  
    "PANOfPOSAgent": "POS1234Z", 
    "OtherDocumentPOSAgent": ""
  },
  "ElectronicInsuranceAccount": {
    "OpenEIA": "Yes",
    "PreferredRepository": "Karvy Insurance Repository Limited",
    "ExistingEIA": "No",
    "ConvertPoliciesToEIA": "Yes"
  },
  "LifeToBeAssured": {
    "MaritalStatus": "Married",
    "ResidentStatus": "Resident",
    "Education": "Post Grad.",
    "Occupation": "Self Employed",
    "OrganisationName": "Doe Designs",
    "FullName": "Jane Doe",
    "IncomeAnnual": "abc123" 
  },
  "PersonalDetailsLifeToBeAssured": {
    "Height": "5'6\"",
    "Weight": "60",
    "DangerousOccupationOrHobbies": "No",
    "EmployedInForces": "No",
    "FamilyMedicalHistory": "No",
    "WeightLoss10KgLast6Months": "No",
    "CongenitalDefect": "No",
    "TestsOrHospitalization": "Yes",
    "InjuryOrMedicalLeave": "No",
    "MedicalConditions": ["Diabetes"],
    "AlcoholConsumption": "No",
    "TobaccoConsumption": "No",
    "NarcoticsConsumption": "No"
  },
  "PreviousPolicyDetails": {
    "PolicyDetails": [
      {
        "Company": "LIC",
        "PolicyNo": "LIC001122",
        "BasicSumAssured": "500000",
        "AnnualPremium": "6000",
        "Status": "In Force"
      }
    ],
    "FamilyInsuranceDetails": {
      "HusbandParentOccupation": "Teacher",
      "HusbandParentIncome": "500000"
    }
  },
  "ProductDetails": {
    "Objective": ["Protection"],
    "ProductName": "ICICI Secure Plus",
    "ModalPremium": "Ten Thousand", 
    "PremiumPaymentTerm": "10",
    "PolicyTerm": "20",
    "SumAssured": "2000000",
    "GMB_GSB": "2050000",
    "Riders": [],
    "BenefitPayoutOption": "Lump sum",
    "AccidentalDeathBenefit": "Yes",
    "AccidentalDeathBenefitPeriod": "20",
    "AcceleratedCriticalIllnessBenefit": "No"
  },
  "NomineeDetails": {
    "FullName": "Jake Doe",
    "DOB": "32-12-2010", 
    "Gender": "Male",
    "Relationship": "Son"
  },
  "AppointeeDetails": {
    "FullName": "",
    "DOB": "",
    "Gender": "",
    "Relationship": ""
  },
  "FirstPremiumDeposit": {
    "Mode": "Cash",
    "Amount": "20000",
    "ChequeNo": "N/A",
    "Bank": "",
    "ThirdPartyPayment": "No",
    "SourceOfFunds": "Salary"
  },
  "PayoutMode": {
    "Mode": "Direct Credit",
    "BankName": "HDFC Bank",
    "BankBranch": "Connaught Place",
    "AccountType": "Savings",
    "AccountNumber": "000123456789",
    "MICRCode": "",
    "IFSCCode": "HDFC0000123"
  },
  "DeclarationInVernacular": {
    "DeclarantName": ""
  },
  "AdvisorConfidentialReport": {
    "NatureOfWork": "Retail Sales",
    "RelationshipWithProposer": "Customer",
    "KnownDurationYears": "2",
    "RelatedToProposer": "No",
    "ProposerIncome": "120000",
    "Assets": {
      "House": "Owned",
      "Vehicle": "2 Wheeler"
    },
    "HealthObservations": {
      "PhysicalHandicap": "No",
      "MentalRetardation": "No",
      "IllnessOrSurgery": "No",
      "MedicalInvestigations": "Yes"
    },
    "OtherRisks": "No",
    "OtherMaterialInfo": "",
    "Remarks": ""
  }
}

"""
  
  # print("INVOKING RAG :")
  # response = call_rag(retriever).invoke({'document':document})

  # print(f"RAG Result: {response}")
  # print("Process finished")
  # print("are we ready to launch")
  # DEBUG LOGGING
  print("📄 Document Sample:\n", document[:500])
  print("Building retriever. . . ")
  retriever = build_or_load_retriever()
  rag_chain = call_rag(retriever)
  response = rag_chain.invoke({'document':document})
  print(f'RAG:RESULT: {response}')


if __name__ == "__main__":
    main()