
# COBBOT 🤓 — Codebasics Q&A Chatbot (RAG Application)

COBBOT is a **Retrieval-Augmented Generation (RAG)** based Question & Answer chatbot.
It answers user questions **only from a custom knowledge base**, ensuring accurate and non-hallucinated responses.

This project demonstrates a **complete end-to-end RAG pipeline** using **LangChain (v0.2+ LCEL)**, **FAISS**, **Gemini (Google Generative AI)**, and **Streamlit**.

---

## 📌 What Problem Does This Project Solve?

Large Language Models (LLMs) are powerful but often **hallucinate answers** when asked domain-specific questions.

**COBBOT solves this by:**
- Retrieving relevant information from a trusted dataset (CSV)
- Providing only that retrieved context to the LLM
- Generating answers strictly grounded in the source data

This technique is known as **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Key Features

- 🔍 Semantic search using **FAISS vector database**
- 🧠 Context-aware answers using **Gemini LLM**
- 📄 Knowledge base built from CSV data (Codebasics FAQs)
- 🛡️ Hallucination control (answers only from retrieved context)
- 🖥️ Interactive **Streamlit** web application
- ⚙️ Built using **modern LangChain Runnable (LCEL)** architecture

---

## 🧱 Tech Stack

- **Python**
- **LangChain (v0.2+)**
- **LangChain Community**
- **Google Generative AI (Gemini)**
- **FAISS**
- **Sentence Transformers (MiniLM)**
- **Streamlit**
- **python-dotenv**

---

## 📂 Project Structure

```
QA_RAG/
│
├── main.py                 # Streamlit UI logic
├── helper_code.py          # Vector DB creation & RAG pipeline
├── requirements.txt        # Project dependencies
├── codebasics_faqs.csv     # Knowledge base (source data)
├── faiss_index/            # FAISS index (auto-generated)
└── README.md
```

---

## 🧠 How the Project Works (Step-by-Step)

### 1️⃣ Data Loading
- FAQ data is stored in a CSV file.
- `CSVLoader` loads each row as a document.

### 2️⃣ Embedding Generation
- Each document is converted into a vector using **MiniLM sentence embeddings**.
- These embeddings capture semantic meaning.

### 3️⃣ Vector Storage (FAISS)
- Embeddings are stored in a **FAISS vector index**.
- The index is saved locally for reuse.

### 4️⃣ User Query
- User submits a question via the Streamlit UI.
- The question is embedded using the same embedding model.

### 5️⃣ Context Retrieval
- FAISS retrieves the most relevant documents based on similarity.
- Only relevant context is selected.

### 6️⃣ Answer Generation (RAG)
- Gemini receives:
  - Retrieved context
  - User question
- Prompt enforces:
  - Answer strictly from context
  - Respond with *"I don't know"* if the answer is missing

---

## 🔑 Creating Google API Key (Gemini)

This project uses **Gemini (Google Generative AI)**, which requires a Google API key.

### Step-by-Step Guide

1️⃣ Go to **Google AI Studio**  
👉 https://aistudio.google.com  

2️⃣ Sign in with your Google account

3️⃣ Click **Get API Key** (top-right corner)

4️⃣ Choose:
- An existing Google Cloud project **OR**
- Create a new project (recommended)

5️⃣ Click **Create API Key**

6️⃣ Copy the generated API key

7️⃣ (Optional but recommended) Restrict the key:
- Open API key settings
- Enable **Generative Language API** only

---

## ⚙️ Setup Instructions (Complete Guide)

### 1️⃣ Clone the Repository
```bash
git clone <your-github-repo-url>
cd QA_RAG
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / Mac
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

⚠️ Do NOT push this file to GitHub.  
Add `.env` to your `.gitignore`.

---

## ▶️ Running the Application

```bash
streamlit run main.py
```

### Application Flow
1. Click **Create Knowledgebase**
2. FAISS index is created from CSV data
3. Enter a question related to Codebasics courses
4. View the generated answer

---

## 🔁 Important Notes

- If you upgrade LangChain versions:
  - Delete `faiss_index/`
  - Recreate the knowledge base
- FAISS indexes are **version-sensitive**

---

## 🧪 Example Questions

- *Do you offer a Java course?*
- *What is the duration of the bootcamp?*
- *Are there any prerequisites?*

---

## 📌 Resume-Ready Project Description

> Developed a Retrieval-Augmented Generation (RAG) chatbot using LangChain, FAISS, and Gemini to deliver domain-specific, context-grounded answers through a Streamlit interface.

---

## 🙌 Author

**Divya Gayatri K**  
BSc - Data Science (2023)  
DBA @ LTIMindtree  
Aspiring Data Scientist / Data Analyst  

---

⭐ If you found this project helpful, feel free to star the repository!
