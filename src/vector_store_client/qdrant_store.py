from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
import os
import json


class QdrantManager:
    def __init__(self, collection_name="my_documents", data_path="data/processed", embedding_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.collection_name = collection_name
        self.data_path = data_path
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.qdrant_vector_store = None

    def load_data(self):
        """
        Carga los documentos y embeddings desde disco y los configura para su uso en Qdrant.
        """
        print("Cargando embeddings y chunks desde disco...")
        chunks_path = os.path.join(self.data_path, "chunks.json")

        # Cargar chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunk_dicts = json.load(f)

        # Convertir a objetos Document
        docs = [
            Document(page_content=chunk["page_content"], metadata=chunk.get("metadata", {}))
            for chunk in chunk_dicts
        ]

        # Configurar QdrantVectorStore
        print("Configurando QdrantVectorStore con RetrievalMode.DENSE...")
        self.qdrant_vector_store = QdrantVectorStore.from_documents(
            docs,
            embedding=self.embedding_model,
            location=":memory:",
            collection_name=self.collection_name,
            retrieval_mode=RetrievalMode.DENSE,
        )
        print(f"QdrantVectorStore configurado para la colección '{self.collection_name}'.")

    def dense_search(self, query, top_k=5):
        """
        Realiza una búsqueda densa en Qdrant usando LangChain.

        params:
            - query: Texto de consulta.
            - param top_k: Número de resultados a devolver.
        return: 
            - Resultados de búsqueda.
        """
        if not self.qdrant_vector_store:
            raise ValueError("QdrantVectorStore no está configurado. Ejecuta 'load_data()' primero.")

        print("Realizando búsqueda densa...")
        return self.qdrant_vector_store.similarity_search(query, k=top_k)
