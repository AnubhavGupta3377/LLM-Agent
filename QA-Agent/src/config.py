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
    os.environ["LANGCHAIN_PROJECT"] = "local-llm-rag-llama-3.2"


@dataclass
class RAG_Config:
    ollama_model = "llama3.2:3b-instruct-fp16"
    
    vectordb_urls = [
        "https://en.wikipedia.org/wiki/Batting_average_(cricket)",
        "https://www.forbesindia.com/article/explainers/odi-cricket-world-cup-winners-list/93319/1",
        "https://currentaffairs.adda247.com/cricket-world-cup-winners-list/",
        "https://www.thecricketpanda.com/icc-odi-world-cup-winners-list/",
        "https://www.zapcricket.com/blogs/newsroom/icc-odi-cricket-world-cup-winners-list"
    ]

    verify_prompts = True
    show_steps = True
