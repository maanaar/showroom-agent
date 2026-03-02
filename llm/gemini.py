import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

_llm = None


def get_gemini() -> ChatOpenAI:
    global _llm
    if _llm is None:
        # Gemini Flash via OpenRouter (OpenAI-compatible endpoint)
        _llm = ChatOpenAI(
            model="google/gemini-flash-1.5",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
            max_tokens=1024,
        )
    return _llm
