from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = Flask(__name__)
CORS(app) 

index = faiss.read_index('faiss_index.index')  
with open('document_chunks.npy', 'rb') as f:
    document_texts = np.load(f, allow_pickle=True) 

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def query_knowledge_base(query, top_n=3):
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_n)
    results = [(document_texts[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    top_documents = " ".join([doc for doc, _ in results])
    result = pipe(f"Based on the following documents: {top_documents}\nAnswer the question: {query}")
    return result[0]['generated_text']

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Flask App!"

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')  
    if not query:
        return jsonify({'error': 'No query provided'}), 400  

    answer = query_knowledge_base(query, top_n=3)  # Get the answer from the knowledge base
    return jsonify({'answer': answer})  # Return the answer as JSON

if __name__ == '__main__':
    app.run(debug=True)

