from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.tools import GooglePlacesTool
import faiss
import os
from collections import deque
from typing import Dict, List, Optional, Any
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain_experimental.autonomous_agents import BabyAGI
from langchain.memory import ConversationBufferMemory

import dotenv

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

import tools.googlePlaces
import tools.search

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

# Define your embedding model
embeddings_model = OpenAIEmbeddings()

# Initialize the vectorstore as empty
embedding_size = 1536

index = faiss.IndexFlatL2(embedding_size)

vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})

# Define Memory
memory = ConversationBufferMemory()

# Define Tools
todo_prompt = PromptTemplate.from_template(
    "You are a great travel planner. You should actively ask from customers how many days they would like to visit the country or city they want to visit, how much it will cost, and activities such as sightseeing, activities, shopping, food, etc."
)
todo_chain = LLMChain(llm=OpenAI(temperature=0.6),
                      prompt=todo_prompt, memory=memory)
google_place_tool = GooglePlacesTool()
search_tool = SerpAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search_tool.run,
        description="Useful for searching for up-to-date information needed to answer questions",
    ),
    Tool(
        name="Todo",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a course for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
    Tool(
        name="MAP",
        func=google_place_tool.run,
        description="This is useful when informing the user of the location of a recommended place or calculating the distance",
    ),
]

prompt = ZeroShotAgent.create_prompt(
    tools,
)

llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

agent_executor.run("한국의 삼성역을 여행할 때 1박 2일 동안 할 만한 여행 루트를 추천해줘")
