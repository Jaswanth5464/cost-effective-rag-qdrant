import os
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from bs4 import BeautifulSoup
import markdown
from loguru import logger

class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

class DocumentLoader:
    """
    Parses PDF, HTML, and Markdown files and extracts text along with metadata.
    """
    @staticmethod
    def infer_category(file_path: Path) -> str:
        """
        Infers document category from file name or directory structure.
        Default heuristics: finance, policy, architecture, general.
        """
        name_lower = file_path.name.lower()
        if any(w in name_lower for w in ["finance", "report", "budget", "cost", "revenue"]):
            return "finance"
        elif any(w in name_lower for w in ["policy", "employee", "hr", "rules"]):
            return "policy"
        elif any(w in name_lower for w in ["architecture", "tech", "guide", "system"]):
            return "engineering"
        return "general"

    @classmethod
    def load_pdf(cls, file_path: Path) -> List[Document]:
        documents = []
        try:
            reader = PdfReader(str(file_path))
            category = cls.infer_category(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(file_path),
                                "document_name": file_path.name,
                                "page": page_num,
                                "category": category
                            }
                        )
                    )
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
        return documents

    @classmethod
    def load_html(cls, file_path: Path) -> List[Document]:
        documents = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove scripts & styles
            for element in soup(["script", "style", "nav", "footer"]):
                element.decompose()
                
            text = soup.get_text(separator="\n").strip()
            category = cls.infer_category(file_path)
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "document_name": file_path.name,
                            "page": 1,
                            "category": category
                        }
                    )
                )
        except Exception as e:
            logger.error(f"Error reading HTML file {file_path}: {e}")
        return documents

    @classmethod
    def load_markdown(cls, file_path: Path) -> List[Document]:
        documents = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                md_text = f.read()
            category = cls.infer_category(file_path)
            if md_text.strip():
                documents.append(
                    Document(
                        page_content=md_text,
                        metadata={
                            "source": str(file_path),
                            "document_name": file_path.name,
                            "page": 1,
                            "category": category
                        }
                    )
                )
        except Exception as e:
            logger.error(f"Error reading Markdown file {file_path}: {e}")
        return documents

    @classmethod
    def load_file(cls, file_path: Path) -> List[Document]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return cls.load_pdf(file_path)
        elif suffix in [".html", ".htm"]:
            return cls.load_html(file_path)
        elif suffix in [".md", ".markdown"]:
            return cls.load_markdown(file_path)
        else:
            logger.warning(f"Unsupported file format skipped: {file_path}")
            return []
