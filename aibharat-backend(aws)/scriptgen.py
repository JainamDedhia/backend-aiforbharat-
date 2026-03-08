import os

import boto3




bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

model_id = "amazon.nova-2-lite-v1:0"

def generate_script_nova(user_idea):

    prompt = f"""
Generate a structured reel script.

Format:
HOOK:
MAIN_SCRIPT:
EMOTIONAL_TRIGGER:
CALL_TO_ACTION:

Idea: {user_idea}
"""

    response = bedrock.converse(
        modelId="amazon.nova-2-lite-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ],
        inferenceConfig={
            "maxTokens": 150,
            "temperature": 0.5
        }
    )

    return response["output"]["message"]["content"][0]["text"]

    print(response)