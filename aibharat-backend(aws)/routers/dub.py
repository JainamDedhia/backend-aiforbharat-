# routers/dub.py
import uuid
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from core.config import jobs
from core.auth import get_current_user_optional
from services.dubber import run_dubbing

router = APIRouter(prefix="/api/dub", tags=["dub"])


@router.post("")
async def dub_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: str = Form("hi"),
    subtitle_language: str = Form("none"),  # default "none" so it's always optional
    user_id: str = Depends(get_current_user_optional)
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "uploading", "progress": 5, "user_id": user_id}

    tmp_path = f"/tmp/{job_id}_{file.filename}"
    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # REMOVED: dead `add_captions_bool` line that was crashing here

    background_tasks.add_task(
        run_dubbing, job_id, tmp_path, target_language, subtitle_language, file.filename
    )
    return {"job_id": job_id}


@router.get("/{job_id}")
async def get_dub(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return jobs[job_id]