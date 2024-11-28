import os
from langchain.document_loaders import PyPDFLoader

def fn_load_pdf(folder_path):
    """
    Carga una lista con archivos PDF.

    params: 
        - folder_path: Ruta de la carpeta con documentos PDF.

    return:
        - documents: Lista con documentos PDF.
    """
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents
