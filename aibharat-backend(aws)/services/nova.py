import base64, json, io
from PIL import Image
from core.config import bedrock_client, call_gemini

MODEL_ID = "amazon.nova-pro-v1:0"

def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def call_nova(prompt: str, images: list = None) -> str:
    """Call Nova Pro with vision. No Gemini fallback (Gemini doesn't support images the same way)."""
    content = []
    if images:
        for img in images:
            content.append({
                "image": {
                    "format": "jpeg",
                    "source": {
                        "bytes": _pil_to_base64(img)
                    }
                }
            })
    content.append({"text": prompt})

    body = json.dumps({
        "messages": [{"role": "user", "content": content}],
        "inferenceConfig": {"maxTokens": 4096, "temperature": 0.3}
    })

    response = bedrock_client.invoke_model(
        modelId=MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response['body'].read())
    return result['output']['message']['content'][0]['text'].strip()


def call_nova_text(prompt: str) -> str:
    """
    Call Nova Pro for text-only tasks.
    Falls back to Gemini automatically if Bedrock is unavailable or quota exceeded.
    """
    # ── Try Bedrock Nova first ──
    try:
        content = [{"text": prompt}]
        body = json.dumps({
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {"maxTokens": 4096, "temperature": 0.3}
        })
        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response['body'].read())
        text = result['output']['message']['content'][0]['text'].strip()
        print("[AI] Used: Bedrock Nova Pro")
        return text

    except Exception as bedrock_error:
        print(f"[AI] Bedrock failed: {bedrock_error} — falling back to Gemini")

    # ── Fallback: Gemini ──
    try:
        text = call_gemini(prompt)
        print("[AI] Used: Gemini fallback")
        return text
    except Exception as gemini_error:
        print(f"[AI] Gemini also failed: {gemini_error}")
        raise Exception(f"All AI providers failed. Bedrock: {bedrock_error} | Gemini: {gemini_error}")
