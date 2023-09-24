from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import dotenv
import os
import chainlit as cl
from langchain.vectorstores.base import VectorStore
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings


dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

# Define embedding model
embeddings_model = OpenAIEmbeddings()

# Initialize the vectorstore as empty
embedding_size = 1536

index = faiss.IndexFlatL2(embedding_size)

vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})


template = """Answer the following questions as best you can, but speaking as passionate travel expert. You must include at lest two tourist attraction per day in the travel itinerary. 
Output the final travel itinerary in following format:
Day 1:
    Morning:
        Breakfast:
    Afternoon:
        Lunch:
    Evening:
        Dinner:
Day 2:
    Morning:
        Breakfast:
    Afternoon:
        Lunch:
    Evening:
        Dinner:
Day 3:
    Morning:
        Breakfast:
    Afternoon:
        Lunch:
    Evening:
        Dinner:
...

Also, translate the final version of the travel itinerary into Korean language.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: a detailed day by day final answer to the original input question

Output the final travel itinerary in following format:
Day 1:
    Morning:
        Breakfast:
    Afternoon:
        Lunch:
    Evening:
        Dinner:
Day 2:
    Morning:
        Breakfast:
    Afternoon:
        Lunch:
    Evening:
        Dinner:
Day 3:
    Morning:
        Breakfast:
    Afternoon:
        Lunch:
    Evening:
        Dinner:
...

Begin! Remember to answer as a passionate and informative travel expert when giving your final answer.


Question: {input}
{agent_scratchpad}"""

template_with_history = """Answer the following questions as best you can, but speaking as passionate travel expert. You must include at lest two tourist attraction per day in the travel itinerary.
Also, you need to include at least two restraurants per day that reflects the traveler's preferences or dietary preferences. 

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer as a passionate and informative travel expert when giving your final answer in Korean.

Previous conversation history:
{history}

Question: {input}
{agent_scratchpad}"""
# Set up a prompt template


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(
                f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def search_online(input_text):
    search = DuckDuckGoSearchRun().run(
        f"site:tripadvisor.com things to do{input_text}")
    return search


def search_festivals(input_text):
    search = DuckDuckGoSearchRun().run(
        f"site:tripadvisor.com festivals in{input_text}")
    return search


def search_hotel(input_text):
    search = DuckDuckGoSearchRun().run(f"site:booking.com {input_text}")
    return search


def search_flight(input_text):
    search = DuckDuckGoSearchRun().run(f"site:skyscanner.com {input_text}")
    return search


def search_general(input_text):
    search = DuckDuckGoSearchRun().run(f"{input_text}")
    return search


memory = ConversationBufferWindowMemory(k=2)


def _handle_error(error) -> str:
    return str(error)[:50]


@cl.on_chat_start
def main():
    tools = [

        Tool(
            name="Search general",
            func=search_general,
            description="useful for when you need to answer general travel questions"
        ),
        Tool(
            name="Search tripadvisor",
            func=search_online,
            description="useful for when you need to answer trip plan questions"
        ),
        Tool(
            name="Search booking",
            func=search_hotel,
            description="useful for when you need to answer hotel questions"
        ),
        Tool(
            name="Search festivals",
            func=search_festivals,
            description="useful for when you need to answer festivals in related areas"
        ),
        Tool(
            name="Search flight",
            func=search_flight,
            description="useful for when you need to answer flight questions"
        )

    ]

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )
    prompt_with_history = CustomPromptTemplate(
        template=template_with_history,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps", "history"]
    )
    output_parser = CustomOutputParser()
    # memory = ConversationBufferWindowMemory(k=2)
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo-0613")
    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        vectorstore=vectorstore,
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
        handle_parsing_errors=True
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory)

    # Store the chain in the user session
    cl.user_session.set("agent", agent_executor)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("agent")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    print(res)
    # Do any post processing here

    # Send the response
    await cl.Message(content=res["output"]).send()
