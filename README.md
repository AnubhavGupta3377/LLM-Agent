Based on this [Langgraph tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/#components)

An LLM agent build usig langchain and langgraph, with functinality such as web search and semantic search.

![RAG Agent](images/langgraph_agent.png)

**Tools Used:**
1. [Ollama](https://ollama.com/download)
    - To run LLMs locally
2. [LangChain](https://python.langchain.com/docs/introduction/), [LangGraph](https://langchain-ai.github.io/langgraph/)
    - For building the ai-agent
    - Visualization of the agent
3. [Nomic Embeddings](https://www.nomic.ai/blog/posts/nomic-embed-text-v1)
    - Used for semantic search using vector embeddings 

**How to run:**
1. Install the dependencies
    ```shell
    pip install -r requirements.txt
    ```
2. Download and install [Ollama](https://ollama.com/download)
3. Fetch a model, such as llama3.2 (default in the repo is llama3.2:3b-instruct-fp16) to run the inference locally:
    ```shell
    ollama pull llama3.2
    ```
4. Set Tavily Key (For web search API):
    ```shell
    export $TAVILY_API_KEY = <TAVILY API KEY>
    ```
5. Set langgraph key
    ```shell
    export LANGSMITH_API_KEY = <LANGSMITH API KEY>
    ```
6. Run the agent
    ```shell
    python rag_agent.py
    ```
7. Or run the notebook ```experiment.ipynb``` to see the agent in action
