import os
import threading
import logging
from datetime import datetime
from typing import Optional, List
import requests

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from storage.database import (
    init_db,
    article_count,
    cluster_count,
    fetch_all_clusters,
    get_connection,
    db_execute,
    fetch_rows,
)
from storage.user_profiles import (
    get_or_create_profile,
    log_feedback,
    get_feedback_history,
)
from personalization.feedback_handler import process_feedback
from rag.chain import ask, personalised_ask
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_app")

# Initialize database
init_db()

app = FastAPI(title="Personalized News Summarizer API")

# Thread locks and status tracking for background jobs
ingest_lock = threading.Lock()
process_lock = threading.Lock()

job_status = {
    "ingest": {"status": "idle", "last_run": None, "error": None},
    "process": {"status": "idle", "last_run": None, "error": None}
}

# --- Pydantic Models for API Requests ---

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    user_id: str
    cluster_id: int
    signal: str
    dwell_seconds: Optional[float] = 0.0

# --- Helper to retrieve articles by ID ---

def get_articles_by_ids(ids: List[int]) -> List[dict]:
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with get_connection() as conn:
        cursor = conn.cursor()
        db_execute(
            cursor,
            f"SELECT id, url, title, source, topic, published_at FROM articles WHERE id IN ({placeholders})",
            ids
        )
        rows = fetch_rows(cursor)
    return rows

# --- Background Task Workers ---

def run_ingest_worker():
    global job_status
    if not ingest_lock.acquire(blocking=False):
        logger.warning("Ingestion job already running.")
        return
    
    try:
        job_status["ingest"]["status"] = "running"
        job_status["ingest"]["error"] = None
        
        from ingestion.pipeline import run_pipeline
        results = run_pipeline()
        
        job_status["ingest"]["status"] = "success"
        job_status["ingest"]["last_run"] = datetime.now().isoformat()
        logger.info("Background ingestion completed successfully: %s", results)
    except Exception as e:
        job_status["ingest"]["status"] = "failed"
        job_status["ingest"]["error"] = str(e)
        logger.error("Background ingestion failed: %s", e)
    finally:
        ingest_lock.release()

def run_process_worker():
    global job_status
    if not process_lock.acquire(blocking=False):
        logger.warning("Processing job already running.")
        return
    
    try:
        job_status["process"]["status"] = "running"
        job_status["process"]["error"] = None
        
        from processing.pipeline import run_processing_pipeline
        from storage.vector_store import sync_from_db
        
        results = run_processing_pipeline(force=True)
        sync_from_db()
        
        job_status["process"]["status"] = "success"
        job_status["process"]["last_run"] = datetime.now().isoformat()
        logger.info("Background processing completed successfully: %s", results)
    except Exception as e:
        job_status["process"]["status"] = "failed"
        job_status["process"]["error"] = str(e)
        logger.error("Background processing failed: %s", e)
    finally:
        process_lock.release()

# --- API Endpoints ---

@app.get("/api/stats")
def get_stats():
    # Check Ollama connection
    ollama_online = False
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/", timeout=2)
        ollama_online = resp.status_code == 200
    except Exception:
        pass
        
    db_stats = article_count()
    return {
        "articles_total": db_stats["total"],
        "articles_unprocessed": db_stats["unprocessed"],
        "clusters_total": cluster_count(),
        "ollama_online": ollama_online,
        "ollama_model": OLLAMA_MODEL,
        "jobs": job_status
    }

@app.get("/api/clusters")
def get_clusters():
    try:
        clusters = fetch_all_clusters()
        # Attach article objects to each cluster for convenient frontend display
        for c in clusters:
            c["articles"] = get_articles_by_ids(c["article_ids"])
        return clusters
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
def post_ask(req: QueryRequest):
    try:
        if req.user_id:
            result = personalised_ask(req.query, user_id=req.user_id)
        else:
            result = ask(req.query)
            
        return {
            "query": result.query,
            "answer": result.answer,
            "sources": result.sources,
            "retrieved_count": result.retrieved_count
        }
    except Exception as e:
        logger.error("RAG Ask failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
def post_feedback(req: FeedbackRequest):
    try:
        # feedback signal handling
        result = process_feedback(
            req.user_id, 
            req.cluster_id, 
            req.signal, 
            dwell_seconds=req.dwell_seconds
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/profile/{user_id}")
def get_profile(user_id: str):
    try:
        profile = get_or_create_profile(user_id)
        # Fetch detailed feedback logs as well
        history = get_feedback_history(user_id, limit=20)
        return {
            "user_id": profile["user_id"],
            "weights": profile["weights"],
            "updated_at": profile["updated_at"],
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
def trigger_ingest(background_tasks: BackgroundTasks):
    if ingest_lock.locked():
        return JSONResponse(status_code=409, content={"message": "Ingestion already running."})
    
    background_tasks.add_task(run_ingest_worker)
    return {"message": "Ingestion started in the background."}

@app.post("/api/process")
def trigger_process(background_tasks: BackgroundTasks):
    if process_lock.locked():
        return JSONResponse(status_code=409, content={"message": "Processing already running."})
    
    background_tasks.add_task(run_process_worker)
    return {"message": "Processing and vector store sync started in the background."}

# --- Serve Static Frontend files ---

# Helper to check static dir
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount files inside static dir
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "FastAPI Server Running. Please place index.html in static/ folder."}
