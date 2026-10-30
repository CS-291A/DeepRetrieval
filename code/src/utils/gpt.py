from openai import OpenAI
import pdb
import os

from src.utils.api_keys import load_secret

_client = None


def get_client():
    """Create the OpenAI client on first use: $OPENAI_API_KEY, else ./openai_api.key."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=load_secret('OPENAI_API_KEY', './openai_api.key'))
    return _client


def gpt_chat(model, prompt, seed=44):
    client = get_client()
    
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=4096,
    temperature=0.5,
    logprobs=True
    )
    
    return response.choices[0].message.content

def gpt_chat_35(prompt, seed=44):
    client = get_client()
    
    response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=4096,
    temperature=0.5,
    logprobs=True
    )
    
    return response.choices[0].message.content

if __name__ == '__main__':
    prompt = "What is the ICD-10 code for Diabetes Mellitus?"
    model = "gpt-3.5-turbo"
    response = gpt_chat(model, prompt)
    print(response)