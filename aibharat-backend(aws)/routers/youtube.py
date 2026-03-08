# routers/youtube.py
import os, json, uuid
from fastapi import APIRouter, UploadFile, File, Form, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from core.config import dynamo, schedule_table
from core.auth import get_current_user_optional
from datetime import datetime

router = APIRouter(prefix="/api/youtube", tags=["youtube"])

CLIENT_CONFIG = {
    "web": {
        "client_id": "225681369746-e0rr0669qbagk0ha3n7l65gjtngh26lj.apps.googleusercontent.com",
        "client_secret": "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",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["https://ai-for-bharat-sigma.vercel.app/api/youtube/callback"]
    }
}

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# Store tokens and flows in memory (keyed by user_id)
_tokens = {}
_flows = {}

@router.get("/auth")
async def youtube_auth(user_id: str = Depends(get_current_user_optional)):
    flow = Flow.from_client_config(CLIENT_CONFIG, scopes=SCOPES)
    flow.redirect_uri = "https://ai-for-bharat-sigma.vercel.app/api/youtube/callback"
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        state=user_id
    )
    # Store flow so callback can reuse the same object (fixes code_verifier issue)
    _flows[user_id] = flow
    return {"auth_url": auth_url}

@router.get("/callback")
async def youtube_callback(code: str, state: str = "guest_user"):
    try:
        # Reuse stored flow to preserve code_verifier
        flow = _flows.get(state)
        if not flow:
            flow = Flow.from_client_config(CLIENT_CONFIG, scopes=SCOPES)
            flow.redirect_uri = "https://ai-for-bharat-sigma.vercel.app/api/youtube/callback"

        flow.fetch_token(code=code)
        credentials = flow.credentials
        _tokens[state] = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": list(credentials.scopes) if credentials.scopes else SCOPES,
        }
        _flows.pop(state, None)
        return RedirectResponse(url="https://ai-for-bharat-sigma.vercel.app/?yt_connected=true")
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/status")
async def youtube_status(user_id: str = Depends(get_current_user_optional)):
    return {"connected": user_id in _tokens}

@router.post("/schedule")
async def schedule_video(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(""),
    scheduled_time: str = Form(...),
    privacy: str = Form("private"),
    user_id: str = Depends(get_current_user_optional)
):
    try:
        if user_id not in _tokens:
            return {"status": "error", "message": "YouTube not connected. Please authenticate first."}

        # Save file temporarily
        tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Save schedule to DynamoDB
        schedule_id = str(uuid.uuid4())
        schedule_table.put_item(Item={
            "schedule_id": schedule_id,
            "user_id": user_id,
            "title": title,
            "description": description,
            "scheduled_time": scheduled_time,
            "privacy": privacy,
            "status": "scheduled",
            "platform": "YouTube",
            "filename": file.filename,
            "created_at": datetime.utcnow().isoformat(),
        })

        from google.oauth2.credentials import Credentials

        creds_data = _tokens[user_id]
        credentials = Credentials(
            token=creds_data["token"],
            refresh_token=creds_data["refresh_token"],
            token_uri=creds_data["token_uri"],
            client_id=creds_data["client_id"],
            client_secret=creds_data["client_secret"],
            scopes=creds_data["scopes"],
        )

        youtube = build("youtube", "v3", credentials=credentials)

        body = {
            "snippet": {
                "title": title,
                "description": description,
                "categoryId": "22",
            },
            "status": {
                "privacyStatus": privacy,
                "publishAt": scheduled_time if privacy == "private" else None,
            }
        }
        if not body["status"]["publishAt"]:
            del body["status"]["publishAt"]

        media = MediaFileUpload(tmp_path, chunksize=-1, resumable=True)
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = request.execute()

        schedule_table.update_item(
            Key={"schedule_id": schedule_id},
            UpdateExpression="SET #s = :s, youtube_id = :yt",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": "uploaded", ":yt": response.get("id", "")}
        )

        os.remove(tmp_path)

        return {
            "status": "success",
            "schedule_id": schedule_id,
            "youtube_id": response.get("id"),
            "youtube_url": f"https://youtube.com/watch?v={response.get('id')}",
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/schedules")
async def get_schedules(user_id: str = Depends(get_current_user_optional)):
    try:
        response = schedule_table.query(
            IndexName="user_id-index",
            KeyConditionExpression="user_id = :uid",
            ExpressionAttributeValues={":uid": user_id}
        )
        items = response.get("Items", [])
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return {"status": "success", "schedules": items}
    except Exception as e:
        return {"status": "error", "schedules": [], "message": str(e)}
