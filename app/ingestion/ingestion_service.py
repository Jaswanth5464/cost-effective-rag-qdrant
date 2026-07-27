import os
from pathlib import Path
from typing import Dict, Any, List
from qdrant_client.http.models import PointStruct
from loguru import logger

from app.core.config import settings
from app.ingestion.document_loader import DocumentLoader, Document
from app.ingestion.chunker import TextChunker, Chunk
from app.embeddings.embedding_service import get_embedding_service
from app.retrieval.vector_store import get_vector_store

class IngestionService:
    """
    Orchestrates end-to-end document scanning, parsing, chunking, idempotent deduplication, and vector storage.
    """
    def __init__(self, documents_dir: str = None):
        self.documents_dir = Path(documents_dir or settings.DOCUMENTS_DIR)
        self.chunker = TextChunker()
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()

    def ingest_directory(self) -> Dict[str, Any]:
        """
        Scans document directory, parses files, splits into chunks, deduplicates, and embeds/stores vectors.
        """
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory '{self.documents_dir}' does not exist. Creating it.")
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            return {
                "status": "success",
                "processed_files": 0,
                "total_chunks": 0,
                "new_chunks_inserted": 0,
                "skipped_chunks": 0,
                "message": f"Directory '{self.documents_dir}' created. Please place PDF, HTML, or MD files inside."
            }

        file_paths = [
            p for p in self.documents_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in [".pdf", ".html", ".htm", ".md", ".markdown"]
        ]

        if not file_paths:
            logger.info(f"No matching documents found in '{self.documents_dir}'.")
            return {
                "status": "success",
                "processed_files": 0,
                "total_chunks": 0,
                "new_chunks_inserted": 0,
                "skipped_chunks": 0,
                "message": "No files found to ingest."
            }

        logger.info(f"Found {len(file_paths)} document(s) to process.")
        
        all_documents: List[Document] = []
        for file_path in file_paths:
            docs = DocumentLoader.load_file(file_path)
            all_documents.extend(docs)

        all_chunks: List[Chunk] = self.chunker.split_documents(all_documents)
        total_chunks = len(all_chunks)
        
        if not all_chunks:
            return {
                "status": "success",
                "processed_files": len(file_paths),
                "total_chunks": 0,
                "new_chunks_inserted": 0,
                "skipped_chunks": 0,
                "message": "No chunks generated from documents."
            }

        # Step 3: Check for existing points in Qdrant (Idempotent ingestion)
        all_chunk_ids = [c.chunk_id for c in all_chunks]
        existing_ids = self.vector_store.get_existing_point_ids(all_chunk_ids)

        new_chunks = [c for c in all_chunks if c.chunk_id not in existing_ids]
        skipped_count = len(existing_ids)

        logger.info(f"Total chunks: {total_chunks} | Existing (skipped): {skipped_count} | New to insert: {len(new_chunks)}")

        if not new_chunks:
            return {
                "status": "success",
                "processed_files": len(file_paths),
                "total_chunks": total_chunks,
                "new_chunks_inserted": 0,
                "skipped_chunks": skipped_count,
                "message": "All document chunks already exist in vector store. Ingestion skipped (idempotent)."
            }

        # Step 4: Embed new chunks only
        texts_to_embed = [c.text for c in new_chunks]
        embeddings = self.embedding_service.embed_documents(texts_to_embed)

        # Step 5: Construct PointStruct for Qdrant
        points = []
        for chunk, vector in zip(new_chunks, embeddings):
            payload = chunk.metadata.copy()
            payload["text"] = chunk.text
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=payload
                )
            )

        # Step 6: Batch upsert
        self.vector_store.upsert_points(points)

        return {
            "status": "success",
            "processed_files": len(file_paths),
            "total_chunks": total_chunks,
            "new_chunks_inserted": len(new_chunks),
            "skipped_chunks": skipped_count,
            "message": f"Successfully ingested {len(new_chunks)} new chunks ({skipped_count} skipped duplicates)."
        }
