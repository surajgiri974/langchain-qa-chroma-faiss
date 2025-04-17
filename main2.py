from langchain.chains import RetrievalQA 
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
import requests
from bs4 import BeautifulSoup
import PyPDF2

# --- Helper: Load Text File ---
def load_text_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return Document(page_content=file.read())

# --- Helper: Load PDF File using PyPDF2 ---
def load_pdf(path):
    text = ""
    with open(path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return Document(page_content=text)

# --- Helper: Load URL Content using BeautifulSoup ---
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)
    return Document(page_content=text)

# --- Start timer ---
start_time = time.time()

# --- Embedding Model ---
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  

# --- Load all documents ---
documents = []

# Text files
file_paths = [
    "documents/shubham.txt",
    "documents/Understanding_Payment_Systems.txt",
    "documents/Digital_Payments_and_Mobile_Banking.txt",
    "documents/Security_in_Banking_and_Payments.txt",
    "documents/The Future of Banking and Payments.txt"
]
documents.extend([load_text_file(path) for path in file_paths])

# PDF files
pdf_paths = [
    "documents/upi.pdf",
]
documents.extend([load_pdf(path) for path in pdf_paths])

# URLs
urls = [
    "https://en.wikipedia.org/wiki/Digital_payment",
    "https://sbi.co.in/web/about-us/about-us"
]
documents.extend([extract_text_from_url(url) for url in urls])

# --- Vector DBs ---
chroma_db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="./data"  
)
faiss_db = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

# --- Retrievers ---
retriever = chroma_db.as_retriever(search_kwargs={"k": 1})
retriever1 = faiss_db.as_retriever(search_kwargs={"k": 1})

# --- LLM and QA Chains ---
llm = OllamaLLM(model="mistral")

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
qa_chain1 = RetrievalQA.from_chain_type(llm=llm, retriever=retriever1)

# --- Q&A Loop ---
let = True
while let:
    question = input("\t2: Exit \nEnter Your Question: ")

    if question == "2" or question.lower() == "exit":
        let = False  
    elif question: 
        answer = qa_chain.invoke({"query": question})
        answer1 = qa_chain1.invoke({"query": question})
        
        print("Answer (Chroma):", answer.get("result"))
        print("Answer (FAISS):", answer1.get("result"))
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        start_time = time.time()
    else:
        print("Invalid input. Please enter a valid option.")
