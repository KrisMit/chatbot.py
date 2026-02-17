import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Document Chatbot", layout="wide")
st.header("Document Intelligence Chatbot")

@st.cache_resource
def process_file(_file_bytes):
    if not _file_bytes.startswith(b'%PDF-'):
        st.error("Invalid PDF file.")
        st.stop()
    
    pdf_reader = PdfReader(_file_bytes)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"  # FIXED
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". "],  # FIXED
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Session state for persistence
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload PDF", type="pdf")

if file is not None:
    with st.spinner("Processing..."):
        file_bytes = file.read()
        st.session_state.vector_store = process_file(file_bytes)
    
    st.success(f"Loaded {st.session_state.vector_store.index.ntotal} chunks")
    
    user_question = st.text_input("Ask about the document:", key="question")
    
    if user_question and st.session_state.vector_store:
        with st.spinner("Answering..."):
            docs = st.session_state.vector_store.similarity_search(user_question, k=4)
            
            llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=1500
            )
            
            prompt = ChatPromptTemplate.from_template("""
            Use ONLY the following context to answer the question. 
            If the answer isn't in the context, say "I don't have enough information."
            
            Context: {context}
            
            Question: {question}
            
            Answer: """)
            
            chain = create_stuff_documents_chain(llm, prompt)
            response = chain.invoke({"context": docs, "question": user_question})
        
        st.markdown("### Answer")
        st.write(response["answer"])
        
        with st.expander("Sources"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}:**")
                st.write(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
else:
    st.info("Upload a PDF to start")


