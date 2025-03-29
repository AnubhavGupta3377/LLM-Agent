import json
from langchain_core.messages import HumanMessage, SystemMessage
from prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT, RELEVANCE_SYSTEM_PROMPT, RELEVANCE_USER_PROMPT


def test_router_prompt(llm_model):
    def route_question(question):
        user_prompt_formatted = ROUTER_USER_PROMPT.format(question=question)
        result = llm_model.invoke(
            [SystemMessage(content=ROUTER_SYSTEM_PROMPT)] + [HumanMessage(content=user_prompt_formatted)]
        )
        return json.loads(result.content)['datasource']
    
    route_question("Who is Sachin Tendulkar?") == "vectorstore"
    route_question("How many ODI world cups has India won?") == "vectorstore"
    route_question("What are large language models?") == "websearch"
    route_question("When did world war II end?") == "websearch"


def test_relevance_prompt(llm_model):
    def grade_relevance(document, question):
        user_prompt_formatted = RELEVANCE_USER_PROMPT.format(document=document, question=question)
        result = llm_model.invoke(
            [SystemMessage(content=RELEVANCE_SYSTEM_PROMPT)] + [HumanMessage(content=user_prompt_formatted)]
        )
        return json.loads(result.content)['relevant']
    
    document1 = "Cricket is a bat-and-ball game played between two teams of eleven players on a field\
              at the center of which is a 22-yard pitch with a wicket at each end"
    document2 = "Llama is a large language model developed by Meta AI."
    document3 = "Ma Long is considered one of the greatest table tennis players of all time."
    question1 = "What is cricket?"
    assert grade_relevance(document1, question1) == "yes"
    assert grade_relevance(document2, question1) == "no"
    assert grade_relevance(document3, question1) == "no"


def concatenate_docs(docs):
    return "\n\n".join(doc for doc in docs)
