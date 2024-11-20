from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

class QdrantVectorStore:

    """
    Se define la clase QdrantVectorStore para manejar BD vectorial en Qdrant con soporte para embeddings densos
    generados por HuggingFace.
    """

    def __init__(self, embedding_model, collection_name="demo_collection", vector_size=768, 
                 distance=Distance.COSINE, qdrant_path="/tmp/langchain_qdrant"):
        
        """
        Inicializa el vector store en Qdrant.

        Args:
            embedding_model (HuggingFaceEmbeddings) : Modelo para generar embeddings.
            collection_name (str)                   : Nombre de la colección en Qdrant.
            vector_size (int)                       : Dimensión de los vectores.
            distance (Distance)                     : Métrica de distancia a usar (coseno)
            qdrant_path (str)                       : Ruta local para almacenamiento Qdrant V.S.


        Flujo:
            1.- Conecta con qdrant mediante QdrantClient.
            2.- Elimina la coleccion si existe
            3.- Crea la coleccion si no existe.

        """
        self.client = QdrantClient(path=qdrant_path)

        try:
            # Eliminar la colección solo si es necesario
            if collection_name in [col.name for col in self.client.get_collections().collections]:
                print(f"Colección '{collection_name}' ya existe. No se eliminará.")
            else:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance))
            
            print(f"Colección '{collection_name}' creada con vector_size={vector_size} y distancia={distance}.")

        except Exception as e:
            print(f"Error al crear o verificar la colección: {e}")

        self.collection_name = collection_name
        self.embedding_model = embedding_model

    def add_embeddings(self, texts, embeddings, metadata=None, batch_size=1000):

        """
        Agrega embeddings a la coleccion del vector store en Qdrant, cargando los datos en lotes.

        Args:
            texts (list of str)                     : Lista de textos asociados con los embeddings.
            embeddings (list of list[float])        : Embeddings generados para los textos.
            metadata (list of dict, optional)       : Metadatos opcionales para cada texto.
            batch_size (int)                        : Tamaño del lote para cargar datos en lotes.

        Flujo:
            1.- Se valida la dimensionalidad de text y embeddings tengan la misma longitud
            2.- Se añaden los metadatos, si no inicializa el proceso con listas vacias.
            3.- Se dividen los datos en lotes según el batch_size.
            4.- Para cada lote se seleccionan los textos, embeddings y mnetadatos del rango correspondientes y
                    se crea una lista de documentos para Qdrant.
            5.-. Se suben los lotes a Qdrant.
        
        """

        if len(texts) != len(embeddings):
            raise ValueError("El número de textos y embeddings debe coincidir.")

        # Agregar metadatos si no se proporciona
        if metadata is None:
            metadata = [{} for _ in texts]

        # Dividir los datos en lotes
        total_batches = (len(texts) + batch_size - 1) // batch_size  

        for batch_index in range(total_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, len(texts))
            
            # Lote actual
            batch_texts = texts[start:end]
            batch_embeddings = embeddings[start:end]
            batch_metadata = metadata[start:end]

            # Preparar payloads para Qdrant
            batch_documents = [{"text": batch_texts[i], "metadata": batch_metadata[i]} for i in range(len(batch_texts))]

            # Subir lote a Qdrant
            self.client.upload_collection(
                collection_name=self.collection_name,
                vectors=batch_embeddings,
                payload=batch_documents)
            
            print(f"Lote {batch_index + 1}/{total_batches} subido: {len(batch_embeddings)} embeddings.")

    def search(self, query, top_k=5):

        """
        Se define un metodo de busca en la colección los embeddings más cercanos a la consulta que retorna
        los top_k embeddings más similares a la consulta.

        Args:
            query (str)     : Texto de consulta.
            top_k (int)     : Número de resultados más cercanos a devolver.

        Flujo:
            1.- Se genera un embedding para el texto de la consulta.
            2.- Se realiza la busqueda en Qdrant usando el embedding.
            3.- Se procesan los resultados extrayendo el score y el payload de cada punto.    

        Returns:
            list: Lista de resultados con sus scores y payloads.
        """

        # Generar embedding para la consulta
        embedding = self.embedding_model.embed_query(query)

        # Realizar la búsqueda
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k)

        # Procesar resultados
        return [{"score": res.score, "payload": res.payload} for res in results]

    def update_embeddings(self, ids, new_embeddings, new_payloads):

        """
        Se define un metodo para actualizar los embeddings para puntos existentes en la coleccion.

        Args:
            ids (list)              : IDs de los puntos a actualizar.
            new_embeddings (list)   : Nuevos vectores para los puntos.
            new_payloads (list)     : Nuevos payloads para los puntos.
        """

        points = [
            {"id": id_, "vector": new_embeddings[i], "payload": new_payloads[i]}
            for i, id_ in enumerate(ids)]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points)

        print(f"Actualizados {len(ids)} puntos en la colección '{self.collection_name}'.")