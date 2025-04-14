from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time

start_time = time.time()

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  

documents = []
file_paths = [
    "documents/introduction_to_bank.txt",
    "documents/Understanding_Payment_Systems.txt",
    "documents/Digital_Payments_and_Mobile_Banking.txt",
    "documents/Security_in_Banking_and_Payments.txt",
    "documents/The Future of Banking and Payments.txt"
]
for path in file_paths:
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        documents.append(Document(page_content=content))

chroma_db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="./data"  
)
faiss_db = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

retriever = chroma_db.as_retriever(search_kwargs={"k": 1})
retriever1 = faiss_db.as_retriever(search_kwargs={"k": 1})


llm = OllamaLLM(model="mistral")

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
qa_chain1 = RetrievalQA.from_chain_type(llm=llm, retriever=retriever1)

let = True
while let:
    question = input("\t1: Continue \n\t2: Exit \nEnter Your Question: ")
    
    if question == "2" or question.lower() == "exit":
        let = False  
    elif question: 
        answer = qa_chain.invoke({"query": question})
        answer1 = qa_chain.invoke({"query": question})
        
        # Print the results
        print("Answer:", answer.get("result"))
        print("Answer:", answer1.get("result"))
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        start_time = time.time()
    else:
        print("Invalid input. Please enter a valid option.")


