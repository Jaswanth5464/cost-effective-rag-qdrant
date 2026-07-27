from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from app.api.routes import router
from app.core.config import settings
from app.retrieval.vector_store import get_vector_store

app = FastAPI(
    title="Cost-Effective RAG Service",
    description=(
        "Production-grade, low-cost Retrieval-Augmented Generation API built with Qdrant, "
        "BAAI/bge-small-en-v1.5 embeddings, and Gemini 2.5 Flash LLM."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    logger.info("Starting Cost-Effective RAG FastAPI service...")
    try:
        # Initialize vector store on startup
        vector_store = get_vector_store()
        logger.info("Vector store connection verified.")
    except Exception as e:
        logger.warning(f"Could not pre-connect vector store on startup: {e}")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
