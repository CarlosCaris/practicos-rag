import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

def fn_dense_embeddings(chunks, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
    """
    Genera embeddings densos para los fragmentos de texto.

    params:
        - chunks: Lista de objetos Document con el contenido de texto.
        - model_name: Nombre del modelo HuggingFace para generar embeddings.
    
    return: 
        - Lista de embeddings (arreglos NumPy).
    """
    print(f"Generando embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = [np.array(embedding_model.embed_query(chunk.page_content)) for chunk in chunks]
    return embeddings
