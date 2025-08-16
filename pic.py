# app.py

from flask import Flask, request, jsonify
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import faiss

app = Flask(__name__)

# Directory containing PDF files
PDF_DIRECTORY = 'docs'  # Directory where the PDFs are stored
os.makedirs(PDF_DIRECTORY, exist_ok=True)  # Ensure the directory exists

# Initialize the model and database
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_db = None

# Load PDFs and embed them into the database
def load_and_embed_existing_pdfs():
    documents = []
    for file in os.listdir(PDF_DIRECTORY):
        if file.endswith(r'c:\Users\HP\Desktop\data sets\flipcs.pdf'):
            pdf_path = os.path.join(PDF_DIRECTORY, file)
            loader = PyPDFLoader(pdf_path)
            doc_pages = loader.load()
            documents.extend(doc_pages)
            print(f'Loaded {len(doc_pages)} pages from {file}')
            
    # Embed documents and add to vector database
    embeddings = [model.encode(doc.text) for doc in documents]
    texts = [doc.text for doc in documents]
    embedding_db.add_texts(texts, embeddings=embeddings)
    print("Documents have been embedded and stored in the vector database.")

# Initialize Chroma or FAISS database
def init_vector_database():
    global embedding_db
    embeddings = HuggingFaceEmbeddings(model=model)
    embedding_db = Chroma(embedding_function=embeddings)
    print("Vector database initialized.")
    load_and_embed_existing_pdfs()  # Load existing PDFs

# Route to search in the vector database
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    query_embedding = model.encode(query)
    # Perform similarity search in the vector database
    search_results = embedding_db.similarity_search_by_vector(query_embedding)
    results = [{'text': doc.page_content} for doc in search_results]

    return jsonify(results)

# Initialize vector database and load PDFs on app startup
with app.app_context():
    init_vector_database()

if __name__ == '__main__':
    app.run(debug=True)
