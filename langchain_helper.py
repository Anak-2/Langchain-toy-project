import streamlit as st
import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import SequentialChain
from IPython.display import display

from secret_key import openapi_key

os.environ['OPENAI_API_KEY'] = openapi_key

llm = OpenAI(temperature=0.5)


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
        llm=llm, prompt=prompt_template_menu, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, menu_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )

    response = chain({"cuisine": cuisine})

    return response


# if __name__ == "__main__":
#     print(generate_restaurant_name_and_items("Italian"))
