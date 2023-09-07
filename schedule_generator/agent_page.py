import streamlit as st
from agent import make_schedule

st.title("Make Your Today's Schedule")

job = st.text_input("당신의 직업 또는 꿈은 무엇인가요?")
aim = st.text_area("이번 주 마무리하고 싶은 계획을 적어주세요")
schedule = st.text_area("내일 확정된 스케쥴을 적어주세요 (스케쥴에 반드시 포함)")

if st.button("make", type="primary"):
    st.write("Recommended schedule")
    response = make_schedule(job, aim, schedule)
    st.write(response['created_schedule'])
