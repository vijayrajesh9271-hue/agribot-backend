from pathlib import Path

from rag import init_vector_db, ingest_pdf

print("Initializing vector DB...")

init_vector_db()

kb_path = Path("knowledge_base")

if not kb_path.exists():
    raise FileNotFoundError(f"{kb_path} not found")

pdfs = list(kb_path.glob("*.pdf"))

print(f"Found {len(pdfs)} PDF(s)\n")

for pdf in pdfs:
    print(f"Processing: {pdf.name}")
    ingest_pdf(str(pdf))
    print()

print("✅ RAG setup completed successfully!")