import dotenv
import os
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.schema import SystemMessage
from langchain.prompts import PromptTemplate
from langchain import OpenAI
from googletrans import Translator
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.chains import SequentialChain

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

# llm = OpenAI(temperature=0.7)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, streaming=True)

tools = load_tools(["serpapi", "llm-math"], llm=llm)


def make_schedule(job, aim, schedule):

    make_character_template = PromptTemplate(
        input_variables=[],
        template="""
            당신은 유능한 컨설턴트 역할입니다.\n
            당신에게 드릴 질문은 앞으로 자신이 목표하는 꿈을 위해.\n
            어떻게 하루를 잘 보낼까 고민하는 사람입니다.
        """
    )

    character_chain = LLMChain(
        llm=llm,
        prompt=make_character_template
    )

    search_career_template = PromptTemplate(
        input_variables=['job'],
        template="""
            질문: 저는 {job}이 되고싶습니다.\n
            이 목표를 달성하기 위해 필요한 공부들을 정리해주세요""",
    )

    loadmap_chain = LLMChain(
        llm=llm,
        prompt=search_career_template,
        output_key="loadmap"
    )

    input_schedule_template = PromptTemplate(
        input_variables=['loadmap', 'aim', 'schedule'],
        template="""
            {schedule}은 반드시 포함해주시고, {aim}을 달성하기 위해 시간을 활용해주세요.\n
            제가 추구해야할 방향은 {loadmap} 입니다. 이걸 참고해서 목표 이외에도 하면 좋을 공부나 활동을 스케쥴에 포함해주세요
            스케쥴을 작성할 때 이 시간에 배치한 이유를 적어주세요.\n
            그리고 시간은 오전 9시부터 오후 10시 또는 오후 11시까지 활동하는 전제 하에 작성해주세요.\n
            스케쥴을 작성할 때 이번 주 공부할 목표는 오늘 다 안 끝내도 괜찮으니 무리하게 채워넣지 않아도 됩니다.\n
            스케쥴은 도표 모양으로 정리해서 도표만 알려주세요.
        """
    )

    schedule_chain = LLMChain(
        llm=llm,
        prompt=input_schedule_template,
        output_key="created_schedule"
    )

    chain = SequentialChain(
        chains=[character_chain, loadmap_chain, schedule_chain],
        input_variables=['job', 'aim', 'schedule'],
        output_variables=['created_schedule']
    )

    response = chain({'job': job, 'aim': aim, 'schedule': schedule})

    return response


response = make_schedule(
    "백엔드 개발자",
    "모던 자바 인 액션 10장부터 14장 까지 읽고, 학교 수업 복습, 그리고 Node.js 기초 공부",
    "1시부터 2시 30분까지 물리 수업, 4시부터 7시까지 북한 사회의 이해 수업"
)

print(response['created_schedule'])
