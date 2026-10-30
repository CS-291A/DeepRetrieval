import os
import pdb

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from anthropic import AnthropicBedrock

AWS_DEFAULT_REGION = "us-west-2"
AWS_KEY_FILE = './claude_api_aws.key'

_client = None


def _aws_credentials():
    """$AWS_ACCESS_KEY_ID / $AWS_SECRET_ACCESS_KEY, else the two lines of ./claude_api_aws.key."""
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    if access_key and secret_key:
        return access_key, secret_key
    if os.path.isfile(AWS_KEY_FILE):
        with open(AWS_KEY_FILE, 'r') as f:
            # two lines, one for AWS_ACCESS_KEY_ID, the next for AWS_SECRET_ACCESS_KEY
            keys = [line.strip() for line in f.readlines()]
        if len(keys) >= 2 and keys[0] and keys[1]:
            return keys[0], keys[1]
        raise RuntimeError(f"{os.path.abspath(AWS_KEY_FILE)} must hold two lines: access key id, then secret key")
    raise RuntimeError(
        f"No AWS credential found: set $AWS_ACCESS_KEY_ID and $AWS_SECRET_ACCESS_KEY "
        f"or create {os.path.abspath(AWS_KEY_FILE)}"
    )


def get_client():
    """Create the Bedrock Anthropic client on first use."""
    global _client
    if _client is None:
        access_key, secret_key = _aws_credentials()
        _client = AnthropicBedrock(
            aws_access_key=access_key,
            aws_secret_key=secret_key,
            aws_region=AWS_DEFAULT_REGION
        )
    return _client



def chat_haiku(prompt):

    if isinstance(prompt, str):
        message = [{
            'role': 'user',
            'content': prompt
        }]
    else:
        message = prompt
    
    message = get_client().messages.create(
        temperature=0,
        model="anthropic.claude-3-haiku-20240307-v1:0",
        max_tokens=1024,
        messages=message,
    )
    
    return message.content[0].text

def chat_sonnet(prompt):
    
    if isinstance(prompt, str):
        message = [{
            'role': 'user',
            'content': prompt
        }]
    else:
        message = prompt

    message = get_client().messages.create(
        temperature=0,
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens=1024,
        messages=message,
    )
    
    return message.content[0].text


if __name__ == "__main__":
    prompt = "Write a haiku about the ocean."
    print(chat_haiku(prompt))