import os
import streamlit as st
import pandas as pd
from llama_index.readers.file import CSVReader
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.settings import Settings
from pathlib import Path

# --- UI Config ---
st.set_page_config(page_title="Memory-Enabled RAG App on CSV", layout="wide")
st.title("🧠 Memory-Enabled RAG App on CSV")
st.markdown("Upload a CSV file and start chatting with memory! Follow-up questions are now smarter.")

# --- Upload CSV ---
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

# --- Input Prompt ---
user_query = st.text_input("💬 Enter your question (e.g. 'What is the total revenue in Q1?')")

# --- Proceed if both file and question are provided ---
if uploaded_file and user_query:
    try:
        # Save uploaded file
        with open("uploaded.csv", "wb") as f:
            f.write(uploaded_file.read())
        file_path = Path("uploaded.csv")

        # Load data from CSV
        csv_reader = CSVReader()
        docs = csv_reader.load_data(file_path)

        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

        # Set global settings
        Settings.llm = OpenAI(model="gpt-3.5-turbo")

        # Enable memory
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

        # Build index and chat engine
        index = VectorStoreIndex.from_documents(docs)
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory
        )

        # Get response
        response = chat_engine.chat(user_query)
        st.success(f"✅ Answer: {response.response}")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
