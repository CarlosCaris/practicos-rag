from dataclasses import dataclass

@dataclass
class ChunkEntity:
    """
    ChunkEntity is a dataclass that represents a chunk of a document into our architecture.
    
    Attributes:
    content (str): The content of the chunk.
    metadata (dict): The metadata of the chunk. (e.g. page number, etc.)
    """
    content: str
    metadata: dict