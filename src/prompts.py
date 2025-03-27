# System prompt to check whether to use vectorstore or web search for a given question
ROUTER_SYSTEM_PROMPT = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to Cricket and cricket records.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search."""

# User prompt for router
ROUTER_USER_PROMPT = """Here's the user question: {question}

Think carefully, and objectively assess whether the question should be routed to datastore or websearch.

Return JSON with single key, "datasource", that is 'websearch' or 'vectorstore' depending on the question."""


# System prompt to check relevance of retrieved documents
RELEVANCE_SYSTEM_PROMPT = """You are an expert at checking the relevance of a document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""


# User prompt to get the relevance of a document to a question
RELEVANCE_USER_PROMPT = """Here is the retrieved document: \n\n
##Document-Start##
{document}
##Document-End##
\n\n Here is the user question: {question}

Think carefully, and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, relevant, that is either 'yes' or 'no' indicating whether the document contains at least some information that is relevant to the question."""


# Prompt for RAG
RAG_QA_PROMPT = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

##Context-Start##
{context} 
##Context-End##
Think carefully about the above context. 

Now, review the user question: {question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""
