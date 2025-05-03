# import essential libraries
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

st.set_page_config(page_title='Document QA ChatBot',
                   page_icon =':robot_face:',
                   layout='centered',
                   initial_sidebar_state='auto')

# load the environment variables
load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# load the llm model, in this case, we use llama3 model
llm = ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')

# create a prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided text only.
Please provide the most accurate responses based on the question.
If answer cannot find from the context, please reply to the users that the information is not found in the provided documents.

<context>
{context}
<context>
Questions:{input}
"""
)

description = '''
A chatbot designed to answer questions directly from your uploaded documents. 
Utilizing state-of-the-art language models and embeddings, the Document QA ChatBot processes and analyzes your PDFs to provide accurate and context-specific answers. 
Whether you need to extract information from research papers, reports, or any other documents, Document QA ChatBot is here to help with seamless, interactive Q&A capabilities.
'''

# function to clear the session state
def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]

# function to load data, split data into chunks, perform embeddings and store in vector database
def vector_embeddings(file):
    if 'vectors' not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':False})
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        st.session_state.docs = []
        st.session_state.final_documents = []

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        final_documents = st.session_state.text_splitter.split_documents(docs)

        # Append the new documents to the existing ones
        st.session_state.docs.extend(docs)
        st.session_state.final_documents.extend(final_documents)

        # Update the vector store with the new documents
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

st.title('Document QA ChatBot')

st.sidebar.title('Documents Uploader')
st.sidebar.write(description)
file = st.sidebar.file_uploader('Upload your document', accept_multiple_files=False, type=['pdf'])
if file:
    vector_embeddings(file)



# Streamli UI --- clear session state (vector DB)
if st.sidebar.button('Refresh'):
    clear_session_state()

# Streamlit UI --- user and bot conversation boxes
user = st.chat_message('User')
bot = st.chat_message('Assistant')

# Streamlit UI --- for user to input their queries
prompt1 = st.chat_input('Please enter your question:')

# initiate the QA retrieval and provide answer to user
try:
    user.write(f'User: {prompt1}')
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever, document_chain)
    response=retrieval_chain.invoke({'input':prompt1})
    bot.write(f'Bot: {response["answer"]}')
except:
    bot.write('Bot: I will only answer question based on the document uploaded...')
