"""Este modulo contiene funciones para dividir documentos en chunks."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

def chunk_document(document: Document) -> list[Document]:
    """Divide un documento en chunks.
    Args:
        document: Documento a dividir.
    Returns:
        Lista de chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents([document])