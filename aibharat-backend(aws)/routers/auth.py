# routers/auth.py
import hashlib
import secrets
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import boto3
from core.config import dynamo

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Users table
users_table = dynamo.Table('creatormentor-users')

def hash_password(password: str) -> str:
    """Simple SHA-256 hash — good enough for a hackathon demo."""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_token(user_id: str) -> str:
    """Generate a simple session token."""
    raw = f"{user_id}:{secrets.token_hex(32)}"
    return hashlib.sha256(raw.encode()).hexdigest()

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    identifier: str  # email or username
    password: str

@router.post("/signup")
async def signup(req: SignupRequest):
    try:
        # Check if username already exists
        existing_username = users_table.get_item(Key={'user_id': f"u_{req.username.lower()}"}).get('Item')
        if existing_username:
            return {"status": "error", "message": "Username already taken"}

        # Check if email already exists
        email_check = users_table.get_item(Key={'user_id': f"e_{req.email.lower()}"}).get('Item')
        if email_check:
            return {"status": "error", "message": "Email already registered"}

        user_id = f"u_{req.username.lower()}"
        token = generate_token(user_id)
        hashed_pw = hash_password(req.password)

        # Store main user record (keyed by username)
        users_table.put_item(Item={
            'user_id': user_id,
            'username': req.username,
            'email': req.email.lower(),
            'password_hash': hashed_pw,
            'token': token,
        })

        # Store email → user_id lookup record
        users_table.put_item(Item={
            'user_id': f"e_{req.email.lower()}",
            'ref': user_id,
            'token': token,
        })

        return {
            "status": "success",
            "user_id": user_id,
            "username": req.username,
            "email": req.email.lower(),
            "token": token,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/login")
async def login(req: LoginRequest):
    try:
        identifier = req.identifier.strip().lower()
        hashed_pw = hash_password(req.password)

        # Try username first, then email
        if '@' in identifier:
            # It's an email — look up the ref record
            email_record = users_table.get_item(Key={'user_id': f"e_{identifier}"}).get('Item')
            if not email_record:
                return {"status": "error", "message": "Invalid email or password"}
            user_id = email_record.get('ref')
        else:
            user_id = f"u_{identifier}"

        # Fetch actual user record
        user = users_table.get_item(Key={'user_id': user_id}).get('Item')
        if not user:
            return {"status": "error", "message": "Invalid username or password"}

        if user.get('password_hash') != hashed_pw:
            return {"status": "error", "message": "Invalid username or password"}

        # Regenerate token on each login
        token = generate_token(user_id)
        users_table.update_item(
            Key={'user_id': user_id},
            UpdateExpression="SET #t = :t",
            ExpressionAttributeNames={"#t": "token"},
            ExpressionAttributeValues={":t": token}
        )
        # Also update email lookup token
        users_table.update_item(
            Key={'user_id': f"e_{user.get('email', '')}"},
            UpdateExpression="SET #t = :t",
            ExpressionAttributeNames={"#t": "token"},
            ExpressionAttributeValues={":t": token}
        )

        return {
            "status": "success",
            "user_id": user_id,
            "username": user.get('username'),
            "email": user.get('email'),
            "token": token,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/me")
async def get_me(token: str):
    """Verify token and return user info."""
    try:
        # Scan for token — small table so fine for demo
        response = users_table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr('token').eq(token) &
                             boto3.dynamodb.conditions.Attr('username').exists()
        )
        items = response.get('Items', [])
        if not items:
            return {"status": "error", "message": "Invalid token"}
        user = items[0]
        return {
            "status": "success",
            "user_id": user.get('user_id'),
            "username": user.get('username'),
            "email": user.get('email'),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
