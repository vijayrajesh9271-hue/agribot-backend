import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer


# import torch
# torch.set_num_threads(1)
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.enabled = False
import pypdf

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
PDF_DIR = "./knowledge_base"
SIMILARITY_THRESHOLD = 0.4
EMBEDDING_DIM = 384          # all-MiniLM-L6-v2 output size
CHUNK_SIZE = 150             # words per chunk
CHUNK_OVERLAP = 20           # overlapping words between chunks

# Load embedding model once (downloads 80MB first time, then cached)
print("📥 Loading embedding model...")
_embedder = SentenceTransformer("all-MiniLM-L6-v2", device = "GPU" if os.getenv("USE_GPU") == "1" else "cpu")  ###
print("✅ Embedding model ready")
_embedder = None

def get_embedder():
    print("RAG STEP 5")
    global _embedder

    if _embedder is None:
        print("📥 Loading embedding model...")
        print("RAG STEP 6")
        _embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )
        _embedder.max_seq_length = 256
        print("RAG STEP 7")
        print("✅ Embedding model ready")

    return _embedder


#dbnconnection

#####
def _build_sync_url(async_url: str) -> str:
    """Strip query params and asyncpg driver prefix for psycopg2 compatibility"""
    base = async_url.split("?")[0]
    base = base.replace("postgresql+asyncpg://", "postgresql://")
    return base

sync_url = _build_sync_url(DATABASE_URL)
#engine = create_engine(sync_url, connect_args={"sslmode": "require"})
engine = create_engine(
    sync_url,
    connect_args={"sslmode": "require"},
    pool_pre_ping=True,   # tests each connection before use; transparently reconnects if Neon closed it
    pool_recycle=280,     # recycle connections before Neon's idle timeout can kill them
)
SessionLocal = sessionmaker(bind=engine)


# Setup
def init_vector_db():
    """Create the vector table in Neon if it doesn't exist"""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id SERIAL PRIMARY KEY,
                source VARCHAR(255) NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM}),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))

        # Index for fast cosine similarity search
        # NOTE: ivfflat requires at least 1 row to build, we create it conditionally
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes WHERE indexname = 'rag_chunks_embedding_idx'
                ) THEN
                    IF (SELECT COUNT(*) FROM rag_chunks) > 0 THEN
                        CREATE INDEX rag_chunks_embedding_idx
                        ON rag_chunks
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 10);
                    END IF;
                END IF;
            END
            $$;
        """))

        conn.commit()
    print("✅ Vector table ready in Neon PostgreSQL")


# Helpers

def get_embedding(input_text: str) -> list:
    """Convert text to a 384-dim vector using local sentence-transformers (free, no API)"""
    # return _embedder.encode(input_text).tolist()
    # return get_embedder().encode(input_text).tolist()
    return get_embedder().encode(
    input_text,
    # convert_to_numpy=True,
    # device="cpu",
    convert_to_tensor=False,
    normalize_embeddings=False,
    show_progress_bar=False
    ).tolist()

_worker_embedder = None

def _embedding_worker(text, queue):
    try:
        global _worker_embedder

        if _worker_embedder is None:
            print("WORKER START")

            from sentence_transformers import SentenceTransformer

            _worker_embedder = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu"
            )

            print("MODEL LOADED")

        embedding = _worker_embedder.encode(
            text,
            convert_to_tensor=False,
            normalize_embeddings=False,
            show_progress_bar=False
        ).tolist()

        print("WORKER DONE")

        queue.put(embedding)

    except Exception as e:
        import traceback

        traceback.print_exc()

        queue.put(f"ERROR: {str(e)}")


def get_embedding(input_text: str) -> list:
    queue = mp.Queue()

    process = mp.Process(
        target=_embedding_worker,
        args=(input_text, queue)
    )

    process.start()

    process.join(timeout=30)

    if process.is_alive():
        process.terminate()
        raise TimeoutError("Embedding generation timed out")

    if queue.empty():
        raise Exception("Embedding worker failed")

    result = queue.get()

    if isinstance(result, str):
        raise Exception(result)


###############################
import threading

_embedder = None
_embedder_lock = threading.Lock()


def get_embedder():

    global _embedder

    with _embedder_lock:

        if _embedder is None:

            print("📥 Loading embedding model...")

            from sentence_transformers import SentenceTransformer
            import torch

            torch.set_num_threads(1)

            _embedder = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu"
            )

            _embedder.max_seq_length = 256

            print("✅ Embedding model ready")

    return _embedder


def get_embedding(input_text: str) -> list:

    embedder = get_embedder()

    embedding = embedder.encode(
        input_text,
        convert_to_tensor=False,
        normalize_embeddings=False,
        show_progress_bar=False
    )

    return embedding.tolist()

def _to_pgvector(embedding: list) -> str:
    """Convert Python list to pgvector string format: [0.1,0.2,...]"""
    return "[" + ",".join(map(str, embedding)) + "]"


# Ingestion

def ingest_pdf(pdf_path: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Read a PDF → split into word chunks → embed each chunk → store in Neon.
    Safe to re-run: deletes old chunks for this PDF before inserting new ones.
    """
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"

    if not full_text.strip():
        print(f"⚠️ No text extracted from {pdf_path} — is it a scanned/image PDF?")
        return 0

    # Split into overlapping chunks
    words = full_text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    pdf_name = Path(pdf_path).stem
    print(f"🔄 Embedding {len(chunks)} chunks from {pdf_name}...")

    with SessionLocal() as session:
        # Remove old data for this PDF
        session.execute(
            text("DELETE FROM rag_chunks WHERE source = :source"),
            {"source": pdf_name}
        )

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            session.execute(
                text("""
                    INSERT INTO rag_chunks (source, chunk_index, content, embedding)
                    VALUES (:source, :chunk_index, :content, CAST(:embedding AS vector))
                """),
                {
                    "source": pdf_name,
                    "chunk_index": i,
                    "content": chunk,
                    "embedding": _to_pgvector(embedding),
                }
            )

        session.commit()

    print(f"✅ Ingested {len(chunks)} chunks from {pdf_path} → Neon DB")
    return len(chunks)


# Querying

def query_rag(question: str, n_results: int = 3) -> dict:
    """
    Search Neon for chunks semantically similar to the question.

    Returns:
        {
            "found": bool,        # True if best similarity >= SIMILARITY_THRESHOLD
            "context": str,       # top chunks joined together (feed this to the LLM)
            "sources": list,      # PDF filenames that matched
            "best_score": float   # 0.0 – 1.0 (1.0 = perfect match)
        }
    """
    
    print("BEFORE EMBEDDING")

    question_embedding = get_embedding(question)

    print("AFTER EMBEDDING")

    embedding_pgvector = _to_pgvector(question_embedding)

    with SessionLocal() as session:
        count = session.execute(text("SELECT COUNT(*) FROM rag_chunks")).scalar()
        if count == 0:
            return {"found": False, "context": "", "sources": [], "best_score": 0.0}

        results = session.execute(
            text("""
                SELECT
                    content,
                    source,
                    1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
                FROM rag_chunks
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
            """),
            {"embedding": embedding_pgvector, "limit": n_results}
        ).fetchall()

    if not results:
        return {"found": False, "context": "", "sources": [], "best_score": 0.0}

    best_score = float(results[0][2])
    
    all_scores = [round(float(r[2]), 3) for r in results]
    print(f"📊 RAG scores: {all_scores} | threshold: {SIMILARITY_THRESHOLD}")

    score_gap = best_score - float(results[-1][2]) if len(results) > 1 else 0
    found = best_score >= SIMILARITY_THRESHOLD and score_gap >= 0.05

    if found:
        print(f"📚 RAG HIT — best: {best_score}, gap: {round(score_gap, 3)}")
    else:
        print(f"📚 RAG MISS — best: {best_score} (below threshold or gap too small)")

   
    # Filter out chunks whose score drops too far below the best score
    # This removes irrelevant chunks
    CHUNK_RELEVANCE_DROP = 0.08  # max allowed drop from best score
    relevant_results = [
        row for row in results 
        if (best_score - float(row[2])) <= CHUNK_RELEVANCE_DROP
    ]
    
    print(f"📊 Using {len(relevant_results)}/{len(results)} chunks after relevance filter")
    
    context = "\n\n---\n\n".join([row[0] for row in relevant_results])
    sources = list(set(row[1] for row in relevant_results))

    return {
        "found": found,
        "context": context,
        "sources": sources,
        "best_score": round(best_score, 3),
    }


# Data Bulk INgestion

def ingest_all_pdfs():
    """Ingest every PDF found in the knowledge_base/ folder into Neon"""
    os.makedirs(PDF_DIR, exist_ok=True)
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️ No PDFs found in {PDF_DIR}/ — RAG will be skipped until you add PDFs")
        return

    for pdf_path in pdf_files:
        ingest_pdf(str(pdf_path))

    with SessionLocal() as session:
        total = session.execute(text("SELECT COUNT(*) FROM rag_chunks")).scalar()
    print(f"✅ RAG ready — {total} total chunks stored in Neon")


def get_chunk_count() -> int:
    """Return total number of stored chunks (used by /api/rag/status)"""
    try:
        with SessionLocal() as session:
            return session.execute(text("SELECT COUNT(*) FROM rag_chunks")).scalar()
    except Exception:
        return 0