from langchain_huggingface import HuggingFaceEmbeddings
from ..domain import EmbeddingRepository

class HuggingFaceEmbeddingRepository(EmbeddingRepository[HuggingFaceEmbeddings]):
    def __init__(self):
        self.embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

    def embed(self, text: str) -> str:
        return self.embeddings_model.embed_query(text)
    
    def get_model(self)->HuggingFaceEmbeddings:
        return self.embeddings_model
    