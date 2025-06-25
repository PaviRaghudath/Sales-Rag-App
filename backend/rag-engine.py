# rag_engine.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
from preprocessing.data_loader import load_all_data
from preprocessing.feature_engineering import prepare_features

# Step 1: Load & Prepare Data
data = load_all_data()
train_data = data["train"]
train_data = prepare_features(train_data, data["oil"], data["holidays"], data["transactions"], data["stores"])
train_data.dropna(inplace=True)

# Step 2: Create Corpus
train_data['text'] = (
    "Store " + train_data['store_nbr'].astype(str) +
    ", Family: " + train_data['family'].astype(str) +
    ", Sales: " + train_data['sales'].astype(str)
)
corpus = train_data['text'].tolist()

# Step 3: Load Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Create Corpus Embeddings
corpus_embeddings = embedding_model.encode(
    corpus,
    batch_size=64,
    show_progress_bar=True,
    convert_to_tensor=False  # Chroma expects plain lists
)

# Step 5: Initialize ChromaDB
chroma_client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chromadb"  # Change path as needed
))

# Step 6: Set Up Chroma Collection with SentenceTransformer-based function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="sales_data",
    embedding_function=sentence_transformer_ef
)

# Step 7: Populate ChromaDB (if empty)
if collection.count() == 0:
    collection.add(
        documents=corpus,
        ids=[f"doc_{i}" for i in range(len(corpus))],
        metadatas=train_data[['store_nbr', 'family']].to_dict(orient='records')
    )

# Export objects if needed in other modules
__all__ = ["train_data", "embedding_model", "collection"]
