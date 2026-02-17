# PDF Insight AI (RAG Chatbot)

An intelligent Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content. This project demonstrates my ability to build end-to-end AI workflows using Python and modern LLM frameworks.

## Tech Stack & Tools
* **Framework:** Streamlit (UI/Frontend)
* **Orchestration:** LangChain
* **LLM:** OpenAI GPT-3.5 Turbo
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** OpenAI Embeddings
* **IDE:** PyCharm Professional

## Key Features
* **Document Ingestion:** Uses `PyPDF2` to extract and process text from user-uploaded PDFs.
* **Text Chunking:** Implements `RecursiveCharacterTextSplitter` for optimized context window management.
* **Vector Search:** Performs semantic similarity searches to find the most relevant document sections for a user's query.
* **Smart Q&A:** Uses LangChain's `load_qa_chain` to provide accurate, context-aware answers.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install streamlit langchain pypdf2 faiss-cpu openai`
3. Add your OpenAI API key to your local environment/Streamlit secrets.
4. Run the app: `streamlit run chatbot.py`
