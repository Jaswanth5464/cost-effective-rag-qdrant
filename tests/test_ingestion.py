import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.document_loader import DocumentLoader
from app.ingestion.chunker import TextChunker

def test_document_loader_markdown(tmp_path):
    md_file = tmp_path / "architecture_guide.md"
    md_file.write_text("# Test Title\nThis is a test architecture guide for RAG.", encoding="utf-8")
    
    docs = DocumentLoader.load_file(md_file)
    assert len(docs) == 1
    assert "Test Title" in docs[0].page_content
    assert docs[0].metadata["category"] == "engineering"

def test_document_loader_html(tmp_path):
    html_file = tmp_path / "company_policy.html"
    html_file.write_text("<html><body><h1>Policy</h1><p>Remote stipend is $150.</p></body></html>", encoding="utf-8")
    
    docs = DocumentLoader.load_file(html_file)
    assert len(docs) == 1
    assert "Remote stipend is $150" in docs[0].page_content
    assert docs[0].metadata["category"] == "policy"

def test_chunker_idempotency_hashing(tmp_path):
    md_file = tmp_path / "architecture_guide.md"
    md_file.write_text("Acme Corp RAG architecture guide content.", encoding="utf-8")
    
    docs = DocumentLoader.load_file(md_file)
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    chunks1 = chunker.split_documents(docs)
    chunks2 = chunker.split_documents(docs)
    
    assert len(chunks1) > 0
    assert len(chunks1) == len(chunks2)
    # Hashing must be deterministic
    assert chunks1[0].chunk_hash == chunks2[0].chunk_hash
    assert chunks1[0].chunk_id == chunks2[0].chunk_id
