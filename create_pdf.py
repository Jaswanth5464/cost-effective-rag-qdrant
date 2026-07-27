import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

pdf_path = "documents/financial_report.pdf"
os.makedirs("documents", exist_ok=True)

doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
)

styles = getSampleStyleSheet()
normal = styles['Normal']
heading1 = styles['Heading1']
heading2 = styles['Heading2']

body_style = ParagraphStyle(
    'HighPrecisionBodyStyle',
    parent=normal,
    fontSize=10,
    leading=15,
    spaceAfter=10
)

story = []

# --- PAGE 1: FINANCIAL PERFORMANCE & REVENUE METRICS ---
story.append(Paragraph("Acme Corporation FY2025 Financial Report", heading1))
story.append(Spacer(1, 10))

story.append(Paragraph("1. Revenue & Income Summary", heading2))
story.append(Paragraph(
    "Acme Corporation achieved record financial performance in FY2025. Total gross revenue reached $42.5 million, "
    "representing a 28% year-over-year growth compared to $33.2 million in FY2024. "
    "Gross profit margin expanded to 68.2%, up from 62.4% in FY2024. Net profit margin reached 18.5%.",
    body_style
))

story.append(Paragraph("2. Operating Expenses & Vector Database Cost Optimization", heading2))
story.append(Paragraph(
    "Total operating expenses for FY2025 were $24.1 million. Under managed Pinecone vector database pods, vector storage cost $1,450 per month ($17,400 annually). "
    "By transitioning to self-hosted Qdrant nodes in Docker on AWS Graviton, monthly compute costs dropped to $140 per month. "
    "This architectural pivot achieved an annual cost reduction of $120,000, representing a 91.4% cost savings.",
    body_style
))

story.append(Paragraph("3. Research & Development Investment", heading2))
story.append(Paragraph(
    "Research and Development expenditure for FY2025 was $8.2 million, accounting for 19.3% of revenue, "
    "dedicated to low-cost vector search algorithms, BAAI/bge-small-en-v1.5 embeddings, and automated RAG evaluation.",
    body_style
))
story.append(PageBreak())

# --- PAGE 2: BALANCE SHEET & CASH FLOW POSITION ---
story.append(Paragraph("4. Balance Sheet & Cash Flow Summary", heading1))
story.append(Spacer(1, 10))

story.append(Paragraph("4.1 Cash Position & Liquidity", heading2))
story.append(Paragraph(
    "Acme Corporation ended FY2025 with $18.6 million in cash and cash equivalents and zero long-term debt. "
    "Current assets totaled $24.8 million against current liabilities of $5.2 million, resulting in a current ratio of 4.77x.",
    body_style
))

story.append(Paragraph("4.2 Operating Cash Flow & FY2026 Guidance", heading2))
story.append(Paragraph(
    "Net operating cash flow reached $12.4 million in FY2025. Management projects FY2026 revenue guidance between $52 million and $56 million.",
    body_style
))

doc.build(story)
print(f"Generated high-precision 2-page financial PDF at {pdf_path}")
