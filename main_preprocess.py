import os
import json
import spacy
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from src.loaders.load_documents import fn_load_pdf
from src.chunking.semantic import fn_semantic_chunk
from src.embedding.dense_embeddings import fn_dense_embeddings

def configure_qdrant_client(embedding_dim, collection_name):
    print("Configurando cliente de Qdrant...")
    qdrant_client = QdrantClient(":memory:")
    print(f"Creando la colección '{collection_name}'...")
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    return qdrant_client

def load_documents(data_path_folder):
    print("Cargando documentos desde la carpeta...")
    documents = fn_load_pdf(data_path_folder)
    if not documents:
        raise ValueError("No se encontraron documentos para procesar.")
    return documents

def chunk_documents(documents, nlp_model, initial_chunk_size, min_length):
    print("Dividiendo documentos en fragmentos...")
    chunks = fn_semantic_chunk(
        documents, nlp_model=nlp_model, initial_chunk_size=initial_chunk_size, min_length=min_length
    )
    return chunks

def generate_embeddings(chunks):
    print("Generando embeddings...")
    embeddings = fn_dense_embeddings(chunks)
    return embeddings

def save_processed_data(processed_dir, embeddings, chunks):
    print("Guardando embeddings y fragmentos en disco...")
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, "embeddings.npy"), embeddings)
    serializable_chunks = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks
    ]
    with open(os.path.join(processed_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(serializable_chunks, f, ensure_ascii=False, indent=4)
    return serializable_chunks

def index_embeddings(qdrant_client, collection_name, embeddings, chunks):
    print("Indexando embeddings en Qdrant...")
    points = [
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={"page_content": chunk["page_content"], "metadata": chunk["metadata"]},
        )
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

def main():
    DATA_RAW_PATH = "data/raw/"
    COLLECTION_NAME = "my_documents"
    EMBEDDING_DIM = 384
    PROCESSED_DIR = "data/processed"
    NLP_MODEL = spacy.load("es_core_news_sm")
    INITIAL_CHUNK_SIZE = 512
    MIN_LENGTH = 100

    try:
        qdrant_client = configure_qdrant_client(EMBEDDING_DIM, COLLECTION_NAME)
        documents = load_documents(DATA_RAW_PATH)
        chunks = chunk_documents(documents, NLP_MODEL, INITIAL_CHUNK_SIZE, MIN_LENGTH)
        embeddings = generate_embeddings(chunks)
        serializable_chunks = save_processed_data(PROCESSED_DIR, embeddings, chunks)
        index_embeddings(qdrant_client, COLLECTION_NAME, embeddings, serializable_chunks)
        print("Preprocesamiento completado exitosamente.")
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")
        raise

if __name__ == "__main__":
    main()
