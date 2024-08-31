from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import cohere
import os
from dotenv import load_dotenv
from flask import stream_with_context
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
import numpy as np

app = Flask(__name__)
CORS(app)
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(api_key=COHERE_API_KEY)

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

# Global variables to store the vector store, QA pipeline, original documents, and LLM
vector_store = None
qa_pipeline = None
original_documents = None
llm = None

def initialize_llm():
    global llm
    llm = Cohere(client=co, model="command-xlarge-nightly")

initialize_llm()

def load_pdf_documents(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        file.save(temp_file.name)
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    data = loader.load()
    os.unlink(temp_file_path)
    return data

def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter()
    split_docs = text_splitter.split_documents(documents)
    vector = FAISS.from_documents(split_docs, cohere_embeddings)
    return vector

def create_qa_pipeline(vector):
    global llm
    retriever = vector.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

def generate_noise(text, noise_factor=0.1):
    tokens = text.split()
    noisy_tokens = []
    for token in tokens:
        if np.random.rand() < noise_factor:
            noisy_tokens.append(np.random.choice(tokens))
        else:
            noisy_tokens.append(token)
    return ' '.join(noisy_tokens)

def impair_step(vector_store, text_to_forget, noise_factor=0.1):
    forget_embedding = cohere_embeddings.embed_documents([text_to_forget])[0]
    similar_docs = vector_store.similarity_search_by_vector(forget_embedding, k=10)
    
    for doc in similar_docs:
        doc_id = doc.metadata.get('id')
        if doc_id is not None:
            try:
                vector_store.delete([doc_id])
                noisy_content = generate_noise(doc.page_content, noise_factor)
                vector_store.add_texts([noisy_content], metadatas=[doc.metadata])
            except Exception as e:
                print(f"Warning: Failed to process document with id {doc_id}. Error: {str(e)}")
    
    return vector_store

def unlearn_data(vector_store, text_to_forget):
    forget_embedding = cohere_embeddings.embed_documents([text_to_forget])[0]
    similar_docs = vector_store.similarity_search_by_vector(forget_embedding, k=10)
    
    for doc in similar_docs:
        doc_id = doc.metadata.get('id')
        if doc_id is not None:
            try:
                vector_store.delete([doc_id])
            except Exception as e:
                print(f"Warning: Failed to delete document with id {doc_id}. Error: {str(e)}")
    
    return vector_store

def repair_step(vector_store, retain_docs):
    for doc in retain_docs:
        vector_store.add_texts([doc.page_content], metadatas=[doc.metadata])
    return vector_store

def refresh_model():
    global llm, qa_pipeline, vector_store
    # Reinitialize the LLM
    initialize_llm()
    # Recreate the QA pipeline with the updated vector store
    qa_pipeline = create_qa_pipeline(vector_store)

@app.route('/upload', methods=['POST'])
def upload_file():
    global vector_store, qa_pipeline, original_documents
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        original_documents = load_pdf_documents(file)
        vector_store = create_vector_store(original_documents)
        qa_pipeline = create_qa_pipeline(vector_store)
        return jsonify({"message": "File processed successfully"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    global qa_pipeline
    data = request.json
    prompt = data['message']
    
    if qa_pipeline is None:
        return jsonify({"error": "No document has been uploaded yet"}), 400

    def generate():
        result = qa_pipeline({"query": prompt})
        yield result["result"]

    return Response(stream_with_context(generate()), content_type='text/plain')

@app.route('/unlearn', methods=['POST'])
def unlearn():
    global vector_store, qa_pipeline, original_documents
    data = request.json
    text_to_forget = data['text_to_forget']
    
    if vector_store is None or original_documents is None:
        return jsonify({"error": "No document has been uploaded yet"}), 400

    try:
        # Impair step
        vector_store = impair_step(vector_store, text_to_forget)
        
        # Unlearn step
        vector_store = unlearn_data(vector_store, text_to_forget)
        
        # Repair step
        retain_docs = [doc for doc in original_documents if text_to_forget not in doc.page_content]
        vector_store = repair_step(vector_store, retain_docs)
        
        # Refresh the model and update QA pipeline
        refresh_model()
        
        return jsonify({"message": "Data unlearned successfully and model refreshed"}), 200
    except Exception as e:
        print(f"Error during unlearning process: {str(e)}")
        return jsonify({"error": f"Unlearning process failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)