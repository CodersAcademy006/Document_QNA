# app.py

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

# ─── Streamlit Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Document QA ChatBot',
    page_icon=':robot_face:',
    layout='centered',
    initial_sidebar_state='auto'
)

# ─── Load & Inject Secrets ─────────────────────────────────────────────────────
load_dotenv()  # reads .env

# Pull from OS, error out if missing
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("❌ Please set GROQ_API_KEY in your .env and restart.")
    st.stop()
# Inject for Groq client to pick up
os.environ["GROQ_API_KEY"] = groq_key

hf_key = os.getenv("HUGGINGFACE_API_KEY")
if not hf_key:
    st.error("❌ Please set HUGGINGFACE_API_KEY in your .env and restart.")
    st.stop()

# ─── Initialize LLM ────────────────────────────────────────────────────────────
# Note: we do *NOT* pass groq_api_key here!
llm = ChatGroq(model_name="Llama3-8b-8192")

# ─── Prompt Template ───────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided text only.
Please provide the most accurate responses based on the question.
If the answer cannot be found in the context, say: 'The information is not found in the provided documents.'

<context>
{context}
</context>

Question: {input}
""")

# ─── Sidebar Description ───────────────────────────────────────────────────────
description = """
A chatbot designed to answer questions directly from your uploaded documents.  
It processes and analyzes your PDFs to provide accurate, context-specific answers.
"""

# ─── Helpers ───────────────────────────────────────────────────────────────────
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def vector_embeddings(pdf_file):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=200
        )
        st.session_state.docs = []
        st.session_state.final_documents = []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        docs = loader.load()
        chunks = st.session_state.text_splitter.split_documents(docs)

        st.session_state.docs.extend(docs)
        st.session_state.final_documents.extend(chunks)
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )
    except Exception as e:
        st.error(f"Error processing PDF: {e}")

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("📄 Document QA ChatBot")

st.sidebar.title("📎 Upload your PDF")
st.sidebar.write(description)

uploaded = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded:
    vector_embeddings(uploaded)

if st.sidebar.button("🔄 Refresh"):
    clear_session_state()

# Chat interface
user_msg = st.chat_input("Ask a question about your document:")
if user_msg:
    st.chat_message("user").write(user_msg)
    try:
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        qa_chain = create_retrieval_chain(retriever, doc_chain)
        ans = qa_chain.invoke({"input": user_msg})["answer"]
        st.chat_message("assistant").write(ans)
    except Exception as e:
        st.chat_message("assistant").write(
            "I can't answer that right now. Make sure you've uploaded a PDF. " 
            f"(Error: {e})"
        )
        st.error(f"Error: {e}")
    finally:
        st.session_state["last_question"] = user_msg
        st.session_state["last_answer"] = ans
        st.session_state["last_docs"] = st.session_state.docs
