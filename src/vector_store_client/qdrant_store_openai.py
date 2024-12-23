from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import OpenAIEmbeddings

import os
import json


class QdrantManager:
    def __init__(self, collection_name="my_documents", data_path="data/processed", embedding_model_name="text-embedding-3-small"):
        """
        Clase para gestionar Qdrant y realizar búsquedas densas.

        params: 
            - collection_name: Nombre de la colección en Qdrant.
            - data_path: Ruta donde se almacenan los embeddings y chunks.
            - embedding_model_name: Nombre del modelo de embeddings.
        """
        self.collection_name = collection_name
        self.data_path = data_path
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name) 
        
        self.qdrant_vector_store = None
        self.retriever = None

    def load_data(self):
        """
        Carga los documentos y embeddings desde disco y los configura para su uso en Qdrant.
        """

        print("Cargando embeddings y chunks desde disco...")
        
        # Verificar que los datos existan
        chunks_path = os.path.join(self.data_path, "chunks.json")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"El archivo de chunks no existe en la ruta '{chunks_path}'. Ejecuta el preprocesamiento primero.")
        
        # Cargar chunks desde JSON
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunk_dicts = json.load(f)

        # Convertir a objetos Document
        docs = [
            Document(page_content=chunk["page_content"], metadata=chunk.get("metadata", {}))
            for chunk in chunk_dicts
        ]

        print("Configurando QdrantVectorStore con RetrievalMode.DENSE...")
        
        # Configurar QdrantVectorStore
        self.qdrant_vector_store = QdrantVectorStore.from_documents(
            docs,
            embedding=self.embedding_model,
            location=":memory:",
            collection_name=self.collection_name,
            retrieval_mode=RetrievalMode.DENSE,
        )
        print(f"QdrantVectorStore configurado para la colección '{self.collection_name}'.")

        self.retriever = self.qdrant_vector_store.as_retriever()

    def search(self, query, top_k=5):
        """
        Realiza una búsqueda densa en Qdrant usando LangChain.

        params: 
            - query: Texto de consulta.
            - top_k: Número de resultados a devolver.
        return: 
            - Resultados de búsqueda.
        """
        if not self.qdrant_vector_store:
            raise ValueError("QdrantVectorStore no está configurado. Ejecuta 'load_data()' primero.")

        print("Realizando búsqueda densa...")
        return self.qdrant_vector_store.similarity_search_with_score(query, k=top_k)
    