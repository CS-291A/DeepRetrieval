import json
import boto3
import time

from src.utils.api_keys import load_secret

AWS_DEFAULT_REGION = "us-west-2"

_boto3_client = None


def get_boto3_client():
    """Create the Bedrock client on first use: $AWS_ACCESS_KEY_ID / $AWS_SECRET_ACCESS_KEY,
    else ./aws_access.key and ./claude_api_aws.key."""
    global _boto3_client
    if _boto3_client is None:
        _boto3_client = boto3.client("bedrock-runtime",
            region_name=AWS_DEFAULT_REGION, 
            aws_access_key_id=load_secret("AWS_ACCESS_KEY_ID", "./aws_access.key"), 
            aws_secret_access_key=load_secret("AWS_SECRET_ACCESS_KEY", "./claude_api_aws.key")
        )
    return _boto3_client


# model="claude-3-haiku-20240307"
BEDROCK_MODEL_NAME_MAP = {
    "claude": "anthropic.claude-v2:1",  # default to sonnet
    "haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "opus": "anthropic.claude-3-opus-20240229-v1:0"
}


def get_claude_response(llm, prompt, max_tokens=2048, temperature=0, tools=None, tool_choice=None):
    model = BEDROCK_MODEL_NAME_MAP.get(llm)
    if model is None:
        raise ValueError(f"Unknown LLM: {llm}")
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",    
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": temperature,
    }
    if tools:
        request_body["tools"] = tools
    if tool_choice:
        request_body["tool_choice"] = tool_choice

    request_body = json.dumps(request_body)

    response = get_boto3_client().invoke_model(
        body=request_body,
        modelId=model
    )
    response_body = json.loads(response.get('body').read())
    return response_body['content'][0]['text']