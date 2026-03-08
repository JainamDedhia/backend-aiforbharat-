import boto3
import json
from datetime import datetime
from decimal import Decimal
import google.generativeai as genai

AWS_REGION = "ap-south-1"
S3_BUCKET = "aiforbharat-dubbing"
AWS_ACCESS_KEY = "AKIAXUK5NBCHMUVFKZHI"
AWS_SECRET_KEY = "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"
DYNAMO_REGION = "us-east-1"

GEMINI_API_KEYS = [
    "AIzaSyBA4K8mA0VNZMut9QT4KULCFfLEcUko95g",
    "AIzaSyC-z5WUd8vQOCfy_W7SR5BXAtDmPGoP26o",
    "AIzaSyA0Mq-c-7XHAAC-eo7GjPtrEr0lImCDFdc",
    "AIzaSyALy5_JCxk5-nIBw5xGKH6Gzlcgu_y-HvQ",
]

POLLY_VOICES = {
    'hi': ('Kajal', 'neural', 'hi-IN'),
    'en': ('Kajal', 'neural', 'en-IN'),
    'fr': ('Lea', 'neural', 'fr-FR'),

}

# ADD THIS to your EDGE_VOICES dict in core/config.py
# Microsoft Edge TTS voices for languages not covered by Polly

EDGE_VOICES = {
    'hi': 'hi-IN-MadhurNeural',       # Hindi
    'mr': 'mr-IN-AarohiNeural',        # Marathi
    'gu': 'gu-IN-NiranjanNeural',      # Gujarati  ← THIS WAS MISSING
    'bn': 'bn-IN-TanishaaNeural',      # Bengali
    'ta': 'ta-IN-ValluvarNeural',      # Tamil
    'te': 'te-IN-MohanNeural',         # Telugu
    'kn': 'kn-IN-GaganNeural',         # Kannada
    'ml': 'ml-IN-MidhunNeural',        # Malayalam
    'en': 'en-IN-PrabhatNeural',       # English (India)
    'fr': 'fr-FR-HenriNeural',         # French
}

TRANSCRIBE_LANG_OPTIONS = ['hi-IN', 'ta-IN', 'te-IN', 'en-IN', 'en-US', 'fr-FR', 'gu-IN', 'mr-IN', 'pa-IN' ]

s3 = boto3.client('s3', region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
transcribe_client = boto3.client('transcribe', region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
translate_client = boto3.client('translate', region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
polly_client = boto3.client('polly', region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# DynamoDB client
dynamo = boto3.resource(
    'dynamodb',
    region_name=DYNAMO_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)
analyses_table = dynamo.Table('creatormentor-analyses')
profiles_table = dynamo.Table('creatormentor-profiles')

# ── DynamoDB helpers ──

def _floats_to_decimal(obj):
    """DynamoDB doesn't accept float — convert to Decimal recursively."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: _floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_floats_to_decimal(i) for i in obj]
    return obj

def _decimal_to_float(obj):
    """Convert Decimal back to float when reading from DynamoDB."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decimal_to_float(i) for i in obj]
    return obj

def save_analysis(user_id: str, job_id: str, result: dict):
    """Save analysis result to DynamoDB."""
    try:
        item = {
            'user_id': user_id,
            'job_id': job_id,
            'timestamp': datetime.utcnow().isoformat(),
            'score': result.get('score', 0),
            'filename': result.get('filename', ''),
            'duration': result.get('duration', 0),
            'metrics': result.get('metrics', {}),
            'mentorAnalysis': result.get('mentorAnalysis', ''),
            'indiaStrategy': result.get('indiaStrategy', []),
            'energyTimeline': result.get('energyTimeline', []),
            'formatChecks': result.get('formatChecks', []),
            'dropOffMoments': result.get('dropOffMoments', []),
            'isVertical': result.get('isVertical', True),
        }
        analyses_table.put_item(Item=_floats_to_decimal(item))
    except Exception as e:
        print(f"[DynamoDB] save_analysis error: {e}")

def get_analyses(user_id: str):
    """Fetch all analyses for a user, sorted by timestamp desc."""
    try:
        response = analyses_table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('user_id').eq(user_id)
        )
        items = response.get('Items', [])
        items = [_decimal_to_float(i) for i in items]
        items.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return items
    except Exception as e:
        print(f"[DynamoDB] get_analyses error: {e}")
        return []

def get_best_and_latest_analysis(user_id: str):
    """Returns (best_analysis, latest_analysis) for script personalization."""
    try:
        analyses = get_analyses(user_id)
        if not analyses:
            return None, None
        latest = analyses[0]  # already sorted desc by timestamp
        best = max(analyses, key=lambda x: x.get('score', 0))
        return best, latest
    except Exception as e:
        print(f"[DynamoDB] get_best_and_latest error: {e}")
        return None, None

def save_profile(user_id: str, profile: dict):
    """Save or update creator profile."""
    try:
        item = {
            'user_id': user_id,
            'niche': profile.get('niche', ''),
            'style': profile.get('style', ''),
            'audience_age': profile.get('audience_age', ''),
            'language': profile.get('language', 'Hinglish'),
            'platform': profile.get('platform', 'Instagram'),
            'shows_face': profile.get('shows_face', 'Yes always'),
            'updated_at': datetime.utcnow().isoformat(),
        }
        profiles_table.put_item(Item=item)
    except Exception as e:
        print(f"[DynamoDB] save_profile error: {e}")

def get_profile(user_id: str):
    """Fetch creator profile."""
    try:
        response = profiles_table.get_item(Key={'user_id': user_id})
        item = response.get('Item')
        return _decimal_to_float(item) if item else None
    except Exception as e:
        print(f"[DynamoDB] get_profile error: {e}")
        return None

_current_key_index = 0

def get_gemini_model():
    global _current_key_index
    genai.configure(api_key=GEMINI_API_KEYS[_current_key_index])
    return genai.GenerativeModel("gemini-2.5-flash")

def call_gemini(content, retry=True):
    global _current_key_index
    model = get_gemini_model()
    try:
        response = model.generate_content(content)
        return response.text.strip()
    except Exception as e:
        error_str = str(e).lower()
        if ('quota' in error_str or 'limit' in error_str or 'exhausted' in error_str) and retry:
            _current_key_index += 1
            if _current_key_index < len(GEMINI_API_KEYS):
                return call_gemini(content, retry=True)
            raise Exception("All Gemini API keys exhausted")
        raise e

jobs: dict = {}

bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name="us-east-1",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)


# YouTube schedules table
schedule_table = dynamo.Table('creatormentor-schedules')
