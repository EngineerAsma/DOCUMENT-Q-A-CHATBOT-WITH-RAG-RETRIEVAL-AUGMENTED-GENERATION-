import streamlit as st
import sys
import os

# ==========================================
# ADD SRC FOLDER TO PYTHON PATH
# ==========================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from graph import ask_question


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📄",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("📄 Document Q&A Chatbot")

st.write(
    "Ask questions about your document using "
    "Retrieval-Augmented Generation (RAG)."
)


# ==========================================
# QUESTION INPUT
# ==========================================

question = st.text_input(
    "Enter your question:"
)


# ==========================================
# ASK BUTTON
# ==========================================

if st.button("Ask Question"):

    if question.strip():

        with st.spinner("Searching document and generating answer..."):

            answer = ask_question(question)

        st.subheader("Answer")

        st.write(answer)

    else:

        st.warning("Please enter a question.")