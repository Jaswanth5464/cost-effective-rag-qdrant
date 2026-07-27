from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from loguru import logger
from app.core.config import settings

class QdrantVectorStore:
    """
    Qdrant vector database store manager for chunk indexing, metadata filtering, and idempotent vector insertion.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QdrantVectorStore, cls).__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}")
        try:
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
                timeout=2.0
            )
            # Test connectivity
            client.get_collections()
            self.client = client
            self.collection_name = settings.QDRANT_COLLECTION_NAME
            self.ensure_collection_exists()
            logger.info("Connected successfully to Qdrant server.")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant server at {settings.QDRANT_URL}: {e}. Falling back to in-memory Qdrant instance.")
            self.client = QdrantClient(":memory:")
            self.collection_name = settings.QDRANT_COLLECTION_NAME
            self.ensure_collection_exists()

    def ensure_collection_exists(self):
        """
        Creates the Qdrant collection if it does not exist and builds payload indexes for metadata fields.
        """
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                logger.info(f"Creating collection '{self.collection_name}' (dim={settings.EMBEDDING_DIMENSION}, metric=Cosine)")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                # Create payload indices for fast metadata filtering
                self._create_payload_indexes()
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Error checking/creating Qdrant collection: {e}")

    def _create_payload_indexes(self):
        fields = ["category", "source", "document_name"]
        for field in fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.info(f"Created payload index for field '{field}'")
            except Exception as e:
                logger.warning(f"Could not create payload index for '{field}': {e}")

    def get_existing_point_ids(self, point_ids: List[str]) -> set[str]:
        """
        Checks Qdrant for existing point IDs to enable idempotent ingestion.
        Returns a set of point IDs that are ALREADY stored.
        """
        if not point_ids:
            return set()
        
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_payload=False,
                with_vectors=False
            )
            existing = {str(rec.id) for rec in records}
            return existing
        except Exception as e:
            logger.warning(f"Error checking existing points in Qdrant: {e}")
            return set()

    def upsert_points(self, points: List[PointStruct]):
        """
        Upserts a batch of points into Qdrant.
        """
        if not points:
            return
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} vectors into '{self.collection_name}'")

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search with optional metadata filtering.
        """
        qdrant_filter = None
        if metadata_filter:
            conditions = []
            for key, val in metadata_filter.items():
                if val:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=val)
                        )
                    )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True
            )
            search_result = res.points if hasattr(res, "points") else res
        elif hasattr(self.client, "search"):
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True
            )
        else:
            search_result = self.client.search_points(
                collection_name=self.collection_name,
                vector=query_vector,
                filter=qdrant_filter,
                limit=top_k,
                with_payload=True
            ).points

        results = []
        for point in search_result:
            results.append({
                "chunk_id": str(point.id),
                "score": point.score,
                "text": point.payload.get("text", ""),
                "metadata": {
                    "source": point.payload.get("source"),
                    "document_name": point.payload.get("document_name"),
                    "page": point.payload.get("page"),
                    "chunk_number": point.payload.get("chunk_number"),
                    "category": point.payload.get("category"),
                    "chunk_hash": point.payload.get("chunk_hash")
                }
            })
        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """Returns collection stats (points count, status)."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": info.vectors_count or info.points_count,
                "status": info.status.name if hasattr(info.status, 'name') else str(info.status)
            }
        except Exception as e:
            return {"error": str(e)}

def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()
