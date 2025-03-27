Tools Used:
1. [Ollama](https://ollama.com/download)
    - To run LLMs locally
2. [LangChain](https://python.langchain.com/docs/introduction/)
    - For building the ai-agent
3. [Nomic Embeddings](https://www.nomic.ai/blog/posts/nomic-embed-text-v1)
    - Used for semantic search using vector embeddings 

How to run:
1. Download and install [Ollama](https://ollama.com/download)
2. Fetch a model, such as gemma3-4b
    ```shell
    ollama pull gemma3:4b
    ```
3. Install the necessary libraries
    ```shell
    pip install -r requirements.txt
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
