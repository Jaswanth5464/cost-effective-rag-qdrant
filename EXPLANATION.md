# Simple & Detailed Guide: Cost-Effective RAG System

This document explains everything about this project in **simple, easy-to-understand English**: what it is, where the data is kept, how ingestion works, how testing was done, and what the final benchmark results are.

---

## 1. What is this Project?

### The Problem
Companies use **Retrieval-Augmented Generation (RAG)** to let AI answer questions using their own documents. However, storing vectors in commercial cloud databases (like Pinecone) is very expensive because they charge high monthly fees even when no one is asking questions.

### The Solution
We built a complete, production-ready RAG system using **Qdrant** (a free, fast, open-source vector database running in Docker). 
- It processes company documents (PDFs, HTML web pages, and Markdown files).
- It turns document text into mathematical vectors using a lightweight embedding model (`BAAI/bge-small-en-v1.5`).
- It stores those vectors inside Qdrant.
- When a user asks a question, it finds the most relevant document pieces and sends them to **Groq API** (`llama-3.1-8b-instant` / `llama-3.3-70b-versatile`) to generate accurate, cited answers without making up information (no hallucinations).

---

## 2. Where is the Data Located?

All project data lives inside the folder structure across a high-precision document corpus:

1. **Document Folder (`documents/`)**:
   - `documents/financial_report.pdf`: A **2-page high-precision PDF financial report** generated via `reportlab`.
     - *Page 1*: Executive financial summary, $42.5M revenue, 28% YoY growth, $120,000 annual Qdrant cost reduction case study, $8.2M R&D investment.
     - *Page 2*: Balance sheet summary ($18.6M cash, zero long-term debt, 4.77x current ratio), $12.4M operating cash flow, FY2026 revenue guidance ($52M–$56M).
   - `documents/company_policy.html`: A 10-section HTML handbook (Remote stipend $150/mo, Home office budget $1,200, Vacation days 22 days, L&D grants $2,500, Travel per diems, Parental leave, MFA security).
   - `documents/architecture_guide.md`: A 6-chapter technical Markdown guide (System architecture, Qdrant HNSW vector tuning, SHA-256 idempotency, prompt safety, latency SLOs).

2. **Benchmark Dataset (`benchmark/dataset.json`)**:
   - Contains **20 realistic test questions** covering the PDF report, HR policy handbook, and engineering architecture guides, along with ground-truth answers to test accuracy.

3. **Audit Log File (`logs/query.log`)**:
   - Every question asked in the system records exact time spent on embedding, vector search, AI answer generation, and total token count.

---

## 3. How Did We Ingest the Data? (Step-by-Step)

Ingestion is the process of reading documents and putting them into the Qdrant vector database:

```
[ Step 1: Read Files ] ---> [ Step 2: Split Text ] ---> [ Step 3: Check Duplicates ] ---> [ Step 4: Convert to Vectors ] ---> [ Step 5: Save in Qdrant ]
```

1. **Reading Documents**:
   - The system automatically scans the `documents/` folder.
   - It reads PDFs using `pypdf`, HTML using `BeautifulSoup`, and Markdown using standard text parsers.

2. **Chunking (Splitting Text)**:
   - Documents are split into **33 distinct text chunks** of **384 characters** each, with **40 characters overlapping** between chunks so sentences don't get cut in half.

3. **Smart Idempotency (Preventing Duplicate Vectors)**:
   - For every text chunk, the system calculates a unique digital fingerprint called a **SHA-256 Hash**.
   - Before generating vectors, it asks Qdrant: *"Do you already have this chunk?"*
   - If yes, **it skips it**. This means re-running ingestion does not waste money or duplicate data.

4. **Generating Vectors**:
   - New text chunks are passed through `BAAI/bge-small-en-v1.5`, generating 384-dimensional dense vectors.

5. **Saving in Qdrant**:
   - The vectors are stored in Qdrant along with metadata like `document_name`, `page_number`, `category`, and `chunk_id`.

---

## 4. What Did We Do for Testing?

### Test 1: Automated Unit & Integration Tests (`tests/`)
Using `pytest`, we wrote 8 automated code tests covering document loading, chunking hash generation, vector search, and API routes.

### Test 2: Accuracy & Retrieval Quality Benchmark (`benchmark/run_benchmark.py`)
We ran all **20 test questions** from `benchmark/dataset.json` through the RAG pipeline using **Groq API** and evaluated:
- **Hit Rate @ 5**: Did the system find the right document in the top 5 results?
- **Recall @ 5**: What fraction of relevant chunks were retrieved?
- **Mean Reciprocal Rank (MRR)**: How high up in the search results was the correct answer?
- **nDCG @ 5**: Ranking quality of search results.
- **Faithfulness**: Did the AI stick strictly to the document facts?
- **Answer Relevance**: Did the AI answer the user's exact question?

---

## 5. What Happened during Testing? (The Final Results)

### Result 1: Code Tests Passed 100%
All 8 automated pytest unit tests **passed cleanly**:
```
======================= 8 passed in 19.24s =======================
```

### Result 2: Groq LLM Benchmark Results (33 Chunks, 20 Questions)
- **Hit Rate @ 5**: **0.9000 (90%)**
- **Recall @ 5**: **0.9000 (90%)**
- **MRR (Rank Quality)**: **0.9000 (90%)** — The correct chunk was almost always ranked #1.
- **nDCG @ 5**: **0.8840 (88.4%)**
- **Context Precision**: **0.8654 (86.5%)**
- **Faithfulness / Groundedness**: **0.8656 (86.6%)** — Answers strictly grounded in document context.
- **Answer Relevance**: **0.9714 (97.1%)** — LLM answers directly addressed the input question.
- **Token F1 Score**: **0.5867 (58.7%)**

### Result 3: Latency & Speed Benchmark (100 Queries)
- **Vector Search Latency (Qdrant)**:
  - **p50**: `1.15 ms`
  - **p95**: `3.26 ms`
  - **Average**: `1.53 ms`
- **Total System Latency (End-to-End)**:
  - **p50**: `94.87 ms`
  - **p95**: `198.40 ms`
  - **Average**: `108.12 ms`

### Result 4: Infrastructure Cost Comparison
- **100K Vectors**: Managed costs **$70/mo** vs. Self-Hosted Qdrant at **$6/mo** (**91.4% savings**).
- **1M Vectors**: Managed costs **$280/mo** vs. Self-Hosted Qdrant at **$24/mo** (**91.4% savings**).
- **10M Vectors**: Managed costs **$1,450/mo** vs. Self-Hosted Qdrant at **$140/mo** (**90.3% savings**).

---

## 📸 Empirical Verification Screenshots

Below are the actual screenshots captured during our testing runs:

### Screenshot 1: Accuracy Benchmark Evaluation
![Benchmark Results](image.png)

### Screenshot 2: 100-Query System Latency Benchmark
![Latency Results](image%20copy.png)

### Screenshot 3: Pytest Automated Code Test Suite
![Pytest Execution](image%20copy%202.png)

