import streamlit as st
import pandas as pd
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import CSVReader
from pathlib import Path

# Set OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit UI
st.set_page_config(page_title="Memory-Enabled CSV GPT App", layout="wide")
st.title("🧠 Memory-Enabled RAG App on CSV")
st.write("Upload a CSV file and start chatting with memory! Follow-up questions are now smarter.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    with open("uploaded.csv", "wb") as f:
        f.write(uploaded_file.read())
        
    file_path = Path("uploaded.csv")  # <--- Convert to Path object
    # Load data
    csv_reader = CSVReader()
    docs = csv_reader.load_data("uploaded.csv")

    # LLM Setup with Memory
    llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
    embed_model = OpenAIEmbedding(api_key=openai_api_key)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)

    # Memory setup
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    chat_engine = index.as_chat_engine(chat_mode="context", memory=memory)

    # Chat UI
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask a question about your data...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        response = chat_engine.chat(user_input)
        st.session_state.chat_history.append(("assistant", response.response))

    # Display chat
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)
