import dotenv


def env_injection():
    dotenv_file = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenv_file)
