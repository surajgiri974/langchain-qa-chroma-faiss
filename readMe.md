# 🧠 LangChain QA System with Chroma & FAISS

> A document-based Question-Answering system using LangChain, HuggingFace Embeddings, Ollama (Mistral), Chroma, and FAISS — all running locally.

---

## 📄 Overview

This project enables you to ask natural language questions over a set of banking and payment-related documents. It uses:

- 🧩 **LangChain** to orchestrate the chain
- 🧠 **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)
- 🗃️ **Chroma** and **FAISS** as vector databases
- 💬 **Ollama LLM** with the **Mistral** model for generating answers

---

## 📁 Document Files

Your documents go inside a folder named `documents/`:


---

## ⚙️ Setup Instructions

### ✅ Prerequisites

- Python 3.8+
- Ollama with `mistral` model (`ollama run mistral`)
- Rust (needed for Chroma on Windows): https://rustup.rs/
- Git (for cloning)

### 📦 Install Dependencies

```bash
git clone https://github.com/your-username/langchain-qa-chroma-faiss.git
cd langchain-qa-chroma-faiss
pip install -r requirements.txt

python main.py

1: Continue 
2: Exit 
Enter Your Question: What is payment?

Answer: Payment refers to the transfer of money or value from one party to another, typically in exchange for goods, services, or to fulfill obligations.
```
📄 License
This project is licensed under the MIT License.
