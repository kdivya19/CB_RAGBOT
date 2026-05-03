import streamlit as st
from helper_code import vectordb_creation, get_qa_chain

st.title("COBBOT 🤓")

if st.button("Create Knowledgebase"):
    vectordb_creation()
    st.success("Knowledgebase created successfully!")

question = st.text_input("Question:")

if question:
    chain = get_qa_chain()
    response = chain.invoke(question)

    st.header("Answer")
    st.write(response)