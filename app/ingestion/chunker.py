import hashlib
import uuid
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.document_loader import Document
from app.core.config import settings

class Chunk:
    def __init__(self, chunk_id: str, chunk_hash: str, text: str, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.chunk_hash = chunk_hash
        self.text = text
        self.metadata = metadata

class TextChunker:
    """
    Splits Document objects into configurable chunks and generates SHA-256 deterministic UUID v5 IDs for idempotent vector insertion.
    """
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def generate_chunk_hash(self, text: str, source: str, page: int) -> str:
        """
        Creates SHA-256 hash string based on content and source metadata.
        """
        raw = f"{source}::page_{page}::{text.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def hash_to_uuid(self, chunk_hash: str) -> str:
        """
        Converts SHA-256 string to a deterministic UUID v5 compatible with Qdrant Point IDs.
        """
        namespace = uuid.UUID('12345678-1234-5678-1234-567812345678')
        return str(uuid.uuid5(namespace, chunk_hash))

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        global_chunk_count = 1

        for doc in documents:
            sub_texts = self.splitter.split_text(doc.page_content)
            for idx, text in enumerate(sub_texts, start=1):
                c_hash = self.generate_chunk_hash(text, doc.metadata["source"], doc.metadata["page"])
                c_uuid = self.hash_to_uuid(c_hash)
                
                chunk_meta = {
                    "source": doc.metadata["source"],
                    "document_name": doc.metadata["document_name"],
                    "page": doc.metadata["page"],
                    "chunk_number": global_chunk_count,
                    "category": doc.metadata["category"],
                    "chunk_hash": c_hash,
                    "chunk_id": c_uuid,
                    "text": text
                }
                chunks.append(Chunk(chunk_id=c_uuid, chunk_hash=c_hash, text=text, metadata=chunk_meta))
                global_chunk_count += 1

        return chunks
