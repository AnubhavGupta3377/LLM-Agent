import os
import getpass
from dataclasses import dataclass


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


def setup_env():
    _set_env("TAVILY_API_KEY")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    _set_env("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "local-llm-rag-gemma3-4b"


@dataclass
class RAG_Config:
    ollama_model = "gemma3:4b"
    
    vectordb_urls = [
        "https://currentaffairs.adda247.com/cricket-world-cup-winners-list/"
    ]

    verify_prompts = True
