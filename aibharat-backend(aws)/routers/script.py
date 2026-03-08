from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from services.nova import call_nova_text
from core.config import get_best_and_latest_analysis, get_profile, save_profile
from core.auth import get_current_user_optional

router = APIRouter(prefix="/api/script", tags=["script"])

class ScriptRequest(BaseModel):
    idea: str
    duration: str
    tone: Optional[str] = "Same as my style"

class ProfileRequest(BaseModel):
    niche: str
    style: str
    audience_age: str
    language: str
    platform: str
    shows_face: str

@router.post("/generate")
async def generate_script(
    req: ScriptRequest,
    user_id: str = Depends(get_current_user_optional)
):
    try:
        profile = get_profile(user_id)
        best, latest = get_best_and_latest_analysis(user_id)

        if profile:
            profile_context = f"""CREATOR PROFILE:
- Content niche: {profile.get('niche')}
- Speaking style: {profile.get('style')}
- Target audience: {profile.get('audience_age')} age group
- Primary language: {profile.get('language')}
- Platform focus: {profile.get('platform')}
- Shows face on camera: {profile.get('shows_face')}"""
        else:
            profile_context = "CREATOR PROFILE: Not set — generate a general script."

        if best and latest:
            best_is_latest = best.get('job_id') == latest.get('job_id')
            past_context = f"""CREATOR'S PAST REEL DATA (use this to match their style):

Best performing reel (score {best.get('score')}/100):
- Mentor analysis: {str(best.get('mentorAnalysis', ''))[:300]}
- Energy pattern: {best.get('energyTimeline', [])}"""
            if not best_is_latest:
                past_context += f"""

Latest reel (score {latest.get('score')}/100):
- Mentor analysis: {str(latest.get('mentorAnalysis', ''))[:300]}
- Energy pattern: {latest.get('energyTimeline', [])}"""
        elif latest:
            past_context = f"""CREATOR'S LATEST REEL DATA:
- Score: {latest.get('score')}/100
- Mentor analysis: {str(latest.get('mentorAnalysis', ''))[:300]}
- Energy pattern: {latest.get('energyTimeline', [])}"""
        else:
            past_context = "PAST REEL DATA: No previous analyses — generate based on profile only."

        creator_language = profile.get('language', 'Hinglish') if profile else 'Hinglish'

        duration_structures = {
            "15s": "HOOK (0-3s) + ONE MAIN POINT (3-12s) + CTA (12-15s)",
            "30s": "HOOK (0-3s) + ACT 1 (3-15s) + ACT 2 (15-25s) + CTA (25-30s)",
            "60s": "HOOK (0-3s) + ACT 1 (3-20s) + ACT 2 (20-40s) + ACT 3 (40-55s) + CTA (55-60s)",
            "90s": "HOOK (0-3s) + ACT 1 (3-25s) + ACT 2 (25-50s) + ACT 3 (50-75s) + CTA (75-90s)"
        }
        structure = duration_structures.get(req.duration, duration_structures["60s"])

        SCRIPT_PROMPT = f"""You are India's top viral content strategist who writes personalized scripts for creators.
You are NOT writing a generic script — you are writing specifically for THIS creator based on their profile and past performance.

{profile_context}

{past_context}

VIDEO IDEA: {req.idea}
TARGET DURATION: {req.duration}
STRUCTURE TO FOLLOW: {structure}
TONE FOR THIS VIDEO: {req.tone}

SCRIPT WRITING RULES:
- Write in the creator's natural language: {creator_language}
- Match their speaking style exactly from their profile
- Hook MUST be in first 3 seconds — make it a question, shocking statement, or bold claim
- Every section must have a Visual suggestion in brackets on a new line
- CTA must be specific — not just "like and subscribe"
- Make it sound like a REAL person talking, not an AI
- Keep each section tight — no filler words

Write the full script following the structure above.
After the script, on new lines output EXACTLY:

VIRAL_PREDICTION: [score]/100
BEST_PLATFORM: [Instagram Reels or YouTube Shorts]
SUGGESTED_CAPTION: [caption under 150 chars]
SUGGESTED_HASHTAGS: [5-7 hashtags]
PERSONALIZATION_NOTE: [1 sentence explaining how this matches THIS creator's specific style]"""

        result = call_nova_text(SCRIPT_PROMPT)

        return {
            "status": "success",
            "script": result,
            "has_profile": profile is not None,
            "has_past_data": latest is not None,
            "personalization_level": (
                "full" if profile and latest else
                "profile_only" if profile else
                "past_data_only" if latest else
                "generic"
            )
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/profile")
async def save_creator_profile(
    req: ProfileRequest,
    user_id: str = Depends(get_current_user_optional)
):
    try:
        save_profile(user_id, req.dict())
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/profile")
async def get_creator_profile(user_id: str = Depends(get_current_user_optional)):
    try:
        profile = get_profile(user_id)
        return {"status": "success", "profile": profile}
    except Exception as e:
        return {"status": "error", "message": str(e)}
