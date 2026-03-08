# core/auth.py
from fastapi import Header, HTTPException, Query
from typing import Optional

async def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        raise HTTPException(status_code=401, detail="Use 'Bearer <token>'")
    return parts[1]  # just return token as user_id for simple auth

async def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    user_id: Optional[str] = Query(None)
) -> str:
    if user_id:
        return user_id
    if authorization:
        try:
            return await get_current_user(authorization)
        except HTTPException:
            pass
    return "guest_user"