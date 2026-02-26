# import os
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from dotenv import load_dotenv

# load_dotenv()

# _llm = None


# def get_llm() -> ChatHuggingFace:
#     global _llm
#     if _llm is None:
#         endpoint = HuggingFaceEndpoint(
#             repo_id="Qwen/Qwen2.5-7B-Instruct",
#             huggingfacehub_api_token=os.getenv("HF_TOKEN"),
#             provider="hf-inference",
#             max_new_tokens=1024,
#             temperature=0.1,
#             timeout=120,
#         )
#         _llm = ChatHuggingFace(llm=endpoint)
#     return _llm

from langchain_ollama import ChatOllama

_llm = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model="qwen3:4b",
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=1024,
            think=False,  # disable Qwen3 thinking mode (much faster)
        )
    return _llm
