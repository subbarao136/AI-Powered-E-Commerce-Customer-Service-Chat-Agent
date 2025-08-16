import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import faiss

# Directory for documents
PDF_DIRECTORY = 'docs'
os.makedirs(PDF_DIRECTORY, exist_ok=True)

documents = []
for file in os.listdir(PDF_DIRECTORY):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(PDF_DIRECTORY, file))
        documents.extend(loader.load())
    elif file.endswith(".docx") or file.endswith(".doc"):
        loader = Docx2txtLoader(os.path.join(PDF_DIRECTORY, file))
        documents.extend(loader.load())
    elif file.endswith(".txt"):
        loader = TextLoader(os.path.join(PDF_DIRECTORY, file))
        documents.extend(loader.load())

if documents:
    # Embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    document_texts = [doc.page_content for doc in documents]
    document_embeddings = np.array([embedding_model.encode(doc) for doc in document_texts])

    # Create FAISS index
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(document_embeddings)
    faiss.write_index(index, 'faiss_index.index')

    # Save document chunks
    with open('document_chunks.npy', 'wb') as f:
        np.save(f, document_texts)

    print("Index and document chunks have been created successfully.")
else:
    print("No documents found in the 'docs' folder.")
