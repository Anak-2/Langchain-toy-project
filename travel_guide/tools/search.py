import os
import dotenv
from langchain import SerpAPIWrapper

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)


def get_search():
    return SerpAPIWrapper()
