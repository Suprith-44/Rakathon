from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import os
import subprocess
import tempfile
from werkzeug.utils import secure_filename
import pickle
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Cohere
import cohere
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Initialize Firebase Admin SDK
cred = credentials.Certificate("Credentials.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

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

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not all([name, email, password]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Add user data to Firebase Firestore
        db.collection('users').add({
            'name': name,
            'email': email,
            'password': password,
            'createdAt': firestore.SERVER_TIMESTAMP
        })
        return jsonify({"message": "User signed up successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({"success": False, "message": "Missing required fields"}), 400

    try:
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).where('password', '==', password).get()
        if query:
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": False, "message": "Invalid email or password"}), 401
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return jsonify({"message": "Welcome to the dashboard"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process the uploaded file and create the model
        model = create_model(file_path)
        
        # Save the model
        model_filename = f"{filename.rsplit('.', 1)[0]}_model.pkl"
        model_path = os.path.join(MODEL_FOLDER, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return jsonify({
            "message": "File uploaded and model created successfully",
            "model_filename": model_filename
        }), 200
    
    return jsonify({"message": "Invalid file type"}), 400

@app.route('/test_model/<model_filename>', methods=['GET'])
def test_model(model_filename):
    model_path = os.path.join(MODEL_FOLDER, model_filename)
    if not os.path.exists(model_path):
        return jsonify({"message": "Model not found"}), 404

    # Start Streamlit app in a separate process
    streamlit_script = 'path_to_your_streamlit_script.py'
    subprocess.Popen(['streamlit', 'run', streamlit_script, '--', model_path])
    
    return jsonify({"message": "Streamlit app started"}), 200

@app.route('/download_model/<model_filename>', methods=['GET'])
def download_model(model_filename):
    model_path = os.path.join(MODEL_FOLDER, model_filename)
    if not os.path.exists(model_path):
        return jsonify({"message": "Model not found"}), 404
    
    return send_file(model_path, as_attachment=True)

def create_model(file_path):
    # Load PDF documents
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Create vector store
    text_splitter = RecursiveCharacterTextSplitter()
    split_docs = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(split_docs, cohere_embeddings)

    # Create RetrievalQA pipeline
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

if __name__ == '__main__':
    app.run(debug=True)