import streamlit as st
import os
import dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import SequentialChain
from IPython.display import display
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

# model 변수로 지정 가능
llm = OpenAI(temperature=0.5)
memory = ConversationBufferWindowMemory()
chatopenai = ChatOpenAI(model_name="gpt-3.5-turbo")


def generate_restaurant_name_and_items(cuisine):
    # Chain 1: Restaurant Name
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food." +
        "Suggest a fancy name for this "
    )

    name_chain = LLMChain(
        llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    # Chain 2: Restaurant Menu
    prompt_template_menu = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest one menu item for {restaurant_name}."
    )

    menu_chain = LLMChain(
        llm=llm, prompt=prompt_template_menu, output_key="menu_item")

    # Chain 3: Menu Recipe
    prompt_template_recipe = PromptTemplate(
        input_variables=['menu_item'],
        template="Introduce a recipe for {menu_item} and please summarize the {menu_item} recipe in 3 to 5 lines."
    )

    recipe_chain = LLMChain(
        llm=llm, prompt=prompt_template_recipe, output_key="menu_recipe"
    )

    chain = SequentialChain(
        chains=[name_chain, menu_chain, recipe_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_item', 'menu_recipe']
    )

    response = chain({"cuisine": cuisine})

    return response


if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))
