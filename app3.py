import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Cohere
import cohere
import os
from dotenv import load_dotenv
import numpy as np
import tempfile
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Set up Cohere API Key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(COHERE_API_KEY)

# Initialize Cohere client and embedding model
cohere_embeddings = CohereEmbeddings(model="embed-english-v2.0", cohere_api_key=COHERE_API_KEY)

# Define the prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Initialize the Cohere model for LLM
llm = Cohere(client=cohere_client, model="command-xlarge-nightly")

# Function to load PDF documents
def load_pdf_documents(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    data = loader.load()
    os.unlink(temp_file_path)
    return data

# Function to create a VectorStore from documents
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter()
    split_docs = text_splitter.split_documents(documents)
    vector = FAISS.from_documents(split_docs, cohere_embeddings)
    return vector

# Function to create a RetrievalQA pipeline
def create_qa_pipeline(vector):
    retriever = vector.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Unlearning function
def unlearn_data(vector_store, text_to_forget):
    # Convert text_to_forget to embeddings
    forget_embedding = cohere_embeddings.embed_documents([text_to_forget])[0]
    
    # Find similar vectors in the FAISS index
    similar_docs = vector_store.similarity_search_by_vector(forget_embedding, k=10)
    
    # Remove forgotten content from vector store
    for doc in similar_docs:
        vector_store.delete([doc.metadata['id']])
    
    return vector_store

# Noise generation step (simplified)
def generate_noise(text, noise_factor=0.1):
    tokens = text.split()
    noisy_tokens = []
    for token in tokens:
        if np.random.rand() < noise_factor:
            noisy_tokens.append(np.random.choice(tokens))  # Replace with a random token
        else:
            noisy_tokens.append(token)
    return ' '.join(noisy_tokens)

# Impair step
def impair_step(vector_store, text_to_forget, noise_factor=0.1):
    forget_embedding = cohere_embeddings.embed_documents([text_to_forget])[0]
    similar_docs = vector_store.similarity_search_by_vector(forget_embedding, k=10)
    
    for doc in similar_docs:
        noisy_content = generate_noise(doc.page_content, noise_factor)
        vector_store.delete([doc.metadata['id']])
        vector_store.add_texts([noisy_content], metadatas=[doc.metadata])
    
    return vector_store

# Repair step
def repair_step(vector_store, retain_docs):
    for doc in retain_docs:
        vector_store.add_texts([doc.page_content], metadatas=[doc.metadata])
    return vector_store

# Streamlit UI
st.title("Document QA with RAG and Unlearning")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        documents = load_pdf_documents(uploaded_file)
        st.success("Document loaded successfully!")

    vector = create_vector_store(documents)
    st.info("Vector store created.")

    qa_pipeline = create_qa_pipeline(vector)
    st.info("RAG pipeline ready.")

    query = st.text_input("Enter your query:")
    if query:
        with st.spinner("Fetching answer..."):
            result = qa_pipeline({"query": query})
            st.write("Answer:", result["result"])

    st.subheader("Unlearn Data")
    text_to_forget = st.text_area("Enter the text to be forgotten:")
    if st.button("Unlearn"):
        with st.spinner("Unlearning data..."):
            # Noise and Impair steps
            vector = impair_step(vector, text_to_forget)
            
            # Unlearn step
            vector = unlearn_data(vector, text_to_forget)
            
            # Repair step
            retain_docs = [doc for doc in documents if text_to_forget not in doc.page_content]
            vector = repair_step(vector, retain_docs)
            
            qa_pipeline = create_qa_pipeline(vector)
            st.success("Data unlearned successfully!")
            st.info("RAG pipeline updated with unlearned data.")
    
    query2 = st.text_input("Enter your query:", key="query2")
    if query2:
        with st.spinner("Fetching answer..."):
            result = qa_pipeline({"query": query2})
            st.write("Answer:", result["result"])