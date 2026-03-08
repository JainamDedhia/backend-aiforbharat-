# routers/analyze.py
import uuid
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from core.config import jobs, get_analyses
from core.auth import get_current_user_optional
from services.analyzer import run_analysis

router = APIRouter(prefix="/api/analyze", tags=["analyze"])


@router.post("")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_optional)
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 5, "user_id": user_id}

    tmp_path = f"/tmp/{job_id}_{file.filename}"
    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    background_tasks.add_task(run_analysis, job_id, tmp_path, file.filename, user_id)
    return {"job_id": job_id}


@router.get("/history")
async def get_history(user_id: str = "guest_user"):
    """Fetch all analyses for a user from DynamoDB."""
    try:
        analyses = get_analyses(user_id)
        return {"status": "success", "analyses": analyses}
    except Exception as e:
        return {"status": "error", "message": str(e), "analyses": []}


@router.get("/{job_id}")
async def get_analysis(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return jobs[job_id]
