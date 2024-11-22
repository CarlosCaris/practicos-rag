import os
import json
import spacy
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from src.loaders.load_documents import fn_load_pdf
from src.chunking.simple import fn_simple_chunking 
from src.chunking.semantic import fn_semantic_chunk
from src.cleaning.clean_documents import fn_clean_text
from src.embedding.dense_embeddings import fn_dense_embeddings

def preprocess_documents(input_folder, collection_name="my_documents"):
    """
    Preprocesa documentos PDF, genera embeddings y los guarda en archivos para su uso en memoria.

    :param input_folder: Carpeta que contiene los PDFs.
    :param collection_name: Nombre de la colección en Qdrant.
    """
    try:
        # CLIENTE QDRANT
        print("Configurando cliente de Qdrant...")
        qdrant_client = QdrantClient(":memory:")
        embedding_dim = 384

        print(f"Creando la colección '{collection_name}'...")
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

        # LOADING DOCUMENTS
        print("Cargando documentos desde la carpeta...")
        documents = fn_load_pdf(input_folder)
        if not documents:
            raise ValueError("No se encontraron documentos para procesar.")

        # CHUNKING
        print("Dividiendo documentos en fragmentos...")
        chunks = fn_semantic_chunk(documents, nlp_model=spacy.load("es_core_news_sm"), initial_chunk_size=512, min_length=100)

        # CLEANING
        print("Limpiando el contenido de los fragmentos...")
        for chunk in chunks:
            chunk.page_content = fn_clean_text(chunk.page_content)

        # EMBEDDINGS
        embeddings = fn_dense_embeddings(chunks)

        # Convertir los chunks a un formato serializable
        serializable_chunks = [
            {"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks
        ]

        # Guardar embeddings y chunks para referencia futura
        print("Guardando embeddings y fragmentos en disco...")
        processed_dir = "data/processed"
        os.makedirs(processed_dir, exist_ok=True)
        np.save(os.path.join(processed_dir, "embeddings.npy"), embeddings)
        with open(os.path.join(processed_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(serializable_chunks, f, ensure_ascii=False, indent=4)

        # Indexar embeddings en Qdrant
        print("Indexando embeddings en Qdrant...")
        points = [
            PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={"page_content": chunk["page_content"], "metadata": chunk["metadata"]}
            )
            for i, (embedding, chunk) in enumerate(zip(embeddings, serializable_chunks))
        ]
        qdrant_client.upsert(collection_name=collection_name, points=points)

        print("Preprocesamiento completado exitosamente.")
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")
        raise

if __name__ == "__main__":
    input_folder = "data/raw/"
    preprocess_documents(input_folder)
