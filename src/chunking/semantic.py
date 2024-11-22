from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def fn_semantic_chunk(documents, nlp_model, initial_chunk_size=512, min_length=100):
    """
    Realiza una segmentación inicial y luego una segmentación semántica basada en oraciones.

    params:
        - documents: Lista de objetos Document con el atributo `page_content`.
        nlp_model: Modelo spaCy cargado para dividir en oraciones.
        initial_chunk_size: Tamaño máximo de los fragmentos iniciales.
        min_length: Longitud mínima de los fragmentos semánticos.
    
    return: 
        - Lista de objetos Document divididos semánticamente.
    """
    chunks = []

    # Segmentación inicial en fragmentos grandes
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=initial_chunk_size, chunk_overlap=0)
    initial_chunks = []
    
    for document in documents:
        # Crear fragmentos iniciales sin metadata
        splits = text_splitter.create_documents([document.page_content])

        # Añadir metadata manualmente
        for split in splits:
            split.metadata = document.metadata
        
        initial_chunks.extend(splits)

    # Segmentación semántica basada en oraciones
    for chunk in initial_chunks:
        doc = nlp_model(chunk.page_content)
        current_chunk = []
        current_length = 0

        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_length = len(sent_text.split())

            if current_length + sent_length > min_length:
                # Guardar el chunk actual si tiene contenido
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        Document(page_content=chunk_text, metadata=chunk.metadata)
                    )
                current_chunk = [sent_text]
                current_length = sent_length
            else:
                current_chunk.append(sent_text)
                current_length += sent_length

        # Guardar el último chunk si tiene contenido
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Document(page_content=chunk_text, metadata=chunk.metadata)
            )

    return chunks
