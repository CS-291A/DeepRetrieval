import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from src.utils.api_keys import load_secret


def _configure_azure():
    """Export the Azure settings AzureChatOpenAI reads; key from env, else ./openai_api_azure.key."""
    os.environ["AZURE_OPENAI_API_KEY"] = load_secret("AZURE_OPENAI_API_KEY", "./openai_api_azure.key")
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://zifeng-gpt-2.openai.azure.com/"
    os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"




def gpt_chat_35(prompt, query_dict):
    _configure_azure()
    prompt = ChatPromptTemplate.from_template(prompt)
    
    model = AzureChatOpenAI(
        azure_deployment="gpt-35", # "gpt-35"
        model='gpt-35-turbo'
    )
    chain = prompt | model | StrOutputParser()
    
    return chain.invoke(query_dict)

def gpt_chat_35_msg(prompt, max_tokens=None):
    _configure_azure()
    message = HumanMessage(
        content=prompt
    )
    
    if max_tokens is not None:
        model = AzureChatOpenAI(
            azure_deployment="gpt-35", # "gpt-35"
            model='gpt-35-turbo',
            max_completion_tokens=max_tokens
        )
    else:
        model = AzureChatOpenAI(
            azure_deployment="gpt-35", # "gpt-35"
            model='gpt-35-turbo',
        )
    
    response = model.invoke([message]).content
    
    return response
    
def gpt_chat_4(prompt, query_dict):
    _configure_azure()
    prompt = ChatPromptTemplate.from_template(prompt)
    
    model = AzureChatOpenAI(
        azure_deployment="gpt-4", # "gpt-35"
        model='gpt-4'
    )
    chain = prompt | model | StrOutputParser()
    
    return chain.invoke(query_dict)


def gpt_chat_4omini(prompt):
    _configure_azure()
    message = HumanMessage(
        content=prompt
    )
    
    model = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini", # "gpt-35"
        model='gpt-4o-mini'
    )
    response = model.invoke([message]).content
    
    return response

def gpt_chat_4o(prompt, max_tokens=None):
    _configure_azure()
    message = HumanMessage(
        content=prompt
    )
    
    if max_tokens is not None:
        model = AzureChatOpenAI(
            azure_deployment="gpt-4o", # "gpt-35"
            model='gpt-4o',
            max_completion_tokens=max_tokens
        )
    else:
        model = AzureChatOpenAI(
            azure_deployment="gpt-4o", # "gpt-35"
            model='gpt-4o',
        )
    response = model.invoke([message]).content
    
    return response


if __name__ == "__main__":
    temp = gpt_chat_35_msg("Translate this sentence from English to French. I love programming.")
    print(temp)