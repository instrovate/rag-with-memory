import streamlit as st
import pandas as pd
import os
from pathlib import Path
from llama_index.readers.file import CSVReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine

# --- Set up the page ---
st.set_page_config(page_title="Memory-Enabled RAG App on CSV", layout="wide")
st.title("🧠 Memory-Enabled RAG App on CSV")
st.markdown("Upload a CSV file and start chatting with memory! Follow-up questions are now smarter.")

with st.expander("ℹ️ What is a memory-enabled RAG app?"):
    st.markdown("""
    - Regular RAG retrieves and answers only based on your current question.
    - This memory-enabled version remembers your past questions in the session.
    - That means you can follow up with things like “What about the other department?” or “And how about Q3?” — and it still makes sense!
    """)

# --- Upload CSV ---
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

# --- Sample questions ---
with st.expander("💡 Sample Questions You Can Ask"):
    st.markdown("""
    - What’s the average salary of employees in the Sales department?
    - Who has the highest performance score?
    - What are the monthly sales figures for Q2?
    - How many employees are in each department?
    - Which employee has the lowest sales?
    """)

# --- User Question ---
user_query = st.text_input("💬 Ask your question:")

# --- If both uploaded file and question are provided ---
if uploaded_file and user_query:
    try:
        # Save the uploaded file temporarily
        temp_path = Path("uploaded.csv")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Set OpenAI Key from Streamlit secrets
        os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

        # Load CSV using LlamaIndex CSVReader
        csv_reader = CSVReader()
        docs = csv_reader.load_data(temp_path)

        # Set up LLM and Memory
        llm = OpenAI(model="gpt-3.5-turbo")
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        Settings.llm = llm

        # Create Index and Chat Engine
        index = VectorStoreIndex.from_documents(docs)
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
        )

        # Chat!
        response = chat_engine.chat(user_query)
        st.success(f"✅ Answer: {response.response}")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Need a similar AI-powered app for your business? [📞 Contact Us](https://instrovate.com/contact) or [💬 WhatsApp](https://wa.me/917428952788)")
