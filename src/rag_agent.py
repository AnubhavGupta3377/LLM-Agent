from langchain_ollama import ChatOllama
from config import setup_env, RAG_Config
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings

from utils import test_router_prompt, test_relevance_prompt, concatenate_docs
from prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT, RELEVANCE_SYSTEM_PROMPT, RELEVANCE_USER_PROMPT, RAG_QA_PROMPT


setup_env()
rag_config = RAG_Config()

# Chat model for inference
local_llm = rag_config.ollama_model
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

# Sanity check the prompts
if rag_config.verify_prompts:
    test_router_prompt(llm_json_mode)
    test_relevance_prompt(llm_json_mode)
    print("Vefified prompts.")


############## Create vector DB and get corresponding retriever ############
# Load documents
docs = WebBaseLoader(rag_config.vectordb_urls).load()

# Split documents based on number of tokens
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="gpt2",
    chunk_size=1000,
    chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs)

# Add to vectorDB and create retriever
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
)
retriever = vectorstore.as_retriever(k=3)


question = "Which teams have won the ODI world cups and when?"
docs = retriever.invoke(question)
docs_txt = concatenate_docs(docs)
rag_prompt_formatted = RAG_QA_PROMPT.format(context=docs_txt, question=question)
generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
print(generation.content)
