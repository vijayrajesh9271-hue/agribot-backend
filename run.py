
"""Local runner for the Agronomist AI API."""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# VERY IMPORTANT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import uvicorn
from api import app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print("🚀 Starting Agronomist AI API Server...")
    print(f"📡 Server will be available on port {port}")
    print(f"📚 API Documentation: http://localhost:{port}/docs")
    print("⏹️  Press Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )