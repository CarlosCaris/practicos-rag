from langchain.text_splitter import RecursiveCharacterTextSplitter

def fn_simple_chunking(documents, chunk_size=512):
    """
    Divide los documentos en fragmentos usando RecursiveCharacterTextSplitter.

    params: 
        - documents: Lista de documentos (cada uno debe tener el atributo `page_content`).
        - chunk_size: Tamaño máximo de cada fragmento.
    
    return: 
        - Lista de fragmentos divididos.
    """
    chunks = []
    
    for document in documents:
        text = document.page_content
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        
        # Generar fragmentos para el documento actual
        split_recursive_character = text_splitter.create_documents([text])
        
        # Añadir los fragmentos a la lista general
        chunks.extend(split_recursive_character)
    
    return chunks