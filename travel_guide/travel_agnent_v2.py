import json
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import Any, List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import dotenv
import os
import chainlit as cl
from datetime import datetime
from langchain.vectorstores.base import VectorStore
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import GooglePlacesTool
from langchain.output_parsers import CommaSeparatedListOutputParser
from chainlit.input_widget import Select, Switch, Slider


dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

# Define embedding model
embeddings_model = OpenAIEmbeddings()

search_tool = SerpAPIWrapper()

# Initialize the vectorstore as empty
embedding_size = 1536

index = faiss.IndexFlatL2(embedding_size)

vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})

template_with_history = """
Answer the following questions as best you can, but speaking as passionate travel expert.
You must include "airline ticket" and at lest two or more "tourist attraction" per day in the travel itinerary.
Also, you need to include at least two or more "restraurants" per day. 

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
Final Answer: the final answer to the original input question. The final output should be a formatted like markdown as following schema
Remember when getting information about a restaurant and touris attractions and festivals, bring an image and exact address of them that introduces places.
```
# Travel Itinerary 

## Day n
    // you should consider client's Departure date Return date
    
### Tourist Attractions
    // Please write in text why you recommend this place
    // When introducing a location to a client, you should include the actual location and images to explain it.
    // Please write images in markdown format. like following schema
    *** place's name ***
    !["place's name"]("image url must use tool. Get it using the Search image tool")
    
### Restaurants
    // Please write in text why you recommend this place
    // When introducing a location to a client, you should include the actual location and images to explain it.
    // Please write images in markdown format. like following schema
    *** place's name ***
    !["place's name"]("image url must use tool. Get it using the Search image tool")

# Festivals
    // Please write in text why you recommend this place
    // When introducing a location to a client, you should include the actual location and images to explain it.
    // Please write images in markdown format. like following schema
    *** place's name ***
    !["place's name"]("image url must use tool. Get it using the Search image tool")

# Transportation
    // Recommend transportation that is easy to use and well priced
```

Remember when getting information about a restaurant and touris attractions and festivals, bring an image and exact address of them that introduces places.


Previous conversation history:
{history}

Question: {input}
{agent_scratchpad}

Begin! Remember to answer as a passionate and informative travel expert when giving your final answer in Korean.
"""
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
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def search_online(input_text):
    search = search_tool.run(
        f"site:tripadvisor.com things to do{input_text}")
    return search


def search_festivals(input_text):
    search = search_tool.run(
        f"site:tripadvisor.com festivals in{input_text}")
    return search


def search_general(input_text):
    search = DuckDuckGoSearchRun().run(f"{input_text}")
    return search


def search_hotel(input_text):
    search = search_tool.run(f"site:booking.com {input_text}")
    return search


def search_flight(input_text):
    search = search_tool.run(f"site:skyscanner.com {input_text}")
    return search


def search_places(input_text):
    search = GooglePlacesTool().run(f"{input_text}")
    return search


memory = ConversationBufferWindowMemory(k=2)


def _handle_error(error) -> str:
    return str(error)[:50]


dest, start_date, end_date = None, None, None


def _handle_error(error) -> str:
    return str(error)[:50]


@cl.on_chat_start
async def main():

    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k",
                        "gpt-4", "gpt-4-32k"],
                initial_index=0,
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="SAI_Steps",
                label="Stability AI - Steps",
                initial=30,
                min=10,
                max=150,
                step=1,
                description="Amount of inference steps performed on image generation.",
            ),
            Slider(
                id="SAI_Cfg_Scale",
                label="Stability AI - Cfg_Scale",
                initial=7,
                min=1,
                max=35,
                step=0.1,
                description="Influences how strongly your generation is guided to match your prompt.",
            ),
            Slider(
                id="SAI_Width",
                label="Stability AI - Image Width",
                initial=512,
                min=256,
                max=2048,
                step=64,
                tooltip="Measured in pixels",
            ),
            Slider(
                id="SAI_Height",
                label="Stability AI - Image Height",
                initial=512,
                min=256,
                max=2048,
                step=64,
                tooltip="Measured in pixels",
            ),
        ]
    ).send()

    tools = [

        Tool(
            name="Search places",
            func=search_places,
            description="This is useful when search information of real places such as address and place image. You must research within 5 items should not over."
        ),

        Tool(
            name="Search general",
            func=search_general,
            description="useful for when you need to answer general travel questions"
        ),
        Tool(
            name="Search image",
            func=GoogleSerperAPIWrapper(type="images").run,
            description="useful for when you need to get image url"
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
        # Tool(
        #     name="Search flight",
        #     func=search_flight,
        #     description="useful for when you need to answer flight questions"
        # ),
        Tool(
            name="Search festivals",
            func=search_festivals,
            description="useful for when you need to answer festivals in related areas"
        )

    ]

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
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
        handle_parsing_errors=True
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory)

    departure = await cl.AskUserMessage(content="어디서 출발하나요?").send()
    if departure:
        await cl.Message(
            content=f"출발지: {departure['content']}",
        ).send()

    dest = await cl.AskUserMessage(content="어디로 여행하고 싶으세요?").send()
    if dest:
        await cl.Message(
            content=f"목적지: {dest['content']}",
        ).send()

    start_date = await cl.AskUserMessage(content="여행 시작일이 언제인가요? (월/일)", timeout=10).send()
    if start_date:
        await cl.Message(
            content=f"Your travel starts at: {start_date['content']}",
        ).send()

    end_date = await cl.AskUserMessage(content="여행 종료일이 언제인가요? (월/일)", timeout=10).send()
    if end_date:
        await cl.Message(
            content=f"You returns at: {end_date['content']}",
        ).send()

    # Define the date format
    date_format = "%m/%d"
    # Parse the date strings into date objects
    date1 = datetime.strptime(start_date['content'], date_format)
    date2 = datetime.strptime(end_date['content'], date_format)

    date_difference = date2 - date1
    days_difference = date_difference.days + 1

    # Store the chain in the user session
    cl.user_session.set("agent", agent_executor)

    print(dest, start_date, end_date)
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("agent")  # type: LLMChain

    message = f"I want to travel to {dest['content']} from {start_date['content']} to {end_date['content']}. Please plan my trip for me."
    message_kor = f"나는 {start_date['content']}부터 {end_date['content']}까지 ({days_difference}일 동안) {departure['content']}에서 {dest['content']}로 여행하고 싶어. 나 대신에 여행 계획을 짜줘."

    # Call the chain asynchronously
    print(message_kor)
    res = await llm_chain.acall(message_kor, callbacks=[cl.AsyncLangchainCallbackHandler()])
    print(res)
    # Do any post processing here
    print(res['output'])
    # Send the response
    # await cl.Message(content=res["output"]).send()


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)


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
