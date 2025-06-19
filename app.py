import streamlit as st
import pandas as pd
import os
from llama_index.readers.file import CSVReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine

# --- UI Config ---
st.set_page_config(page_title="Memory-Enabled RAG App on CSV", layout="wide")
st.title("🧠 Memory-Enabled RAG App on CSV")
st.markdown("Upload a CSV file and start chatting with memory! Follow-up questions are now smarter.")

with st.expander("💡 What is Memory-Enabled RAG? (Click to expand)"):
    st.markdown("""
    This AI app lets you:
    - Upload any CSV file (up to 200MB)
    - Ask questions in plain English
    - Get answers based on **your data**
    - Ask **follow-up questions** — the chatbot remembers previous context!

    **Use Case**: Want to find insights from employee performance reports, sales data, customer logs? Just upload the file and start chatting!
    """)

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
user_query = st.text_input("💬 Enter your question (e.g. 'What is the total revenue in Q1?')")

if uploaded_file:
    st.markdown("""
    🧠 **Try These Sample Questions**:
    - Who are the top 3 performers in Q2?
    - What is the total sales made by the Marketing department?
    - What was their performance score?
    - How do Sales and HR departments compare in Monthly Sales?
    """)

if uploaded_file and user_query:
    try:
        with open("uploaded.csv", "wb") as f:
            f.write(uploaded_file.read())

        file_path = Path("uploaded.csv")
        csv_reader = CSVReader()
        docs = csv_reader.load_data(file_path)

        # Set up OpenAI LLM
        openai_api_key = st.secrets["openai_api_key"]
        os.environ["OPENAI_API_KEY"] = openai_api_key
        llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        index = VectorStoreIndex.from_documents(docs)
        chat_engine = index.as_chat_engine(chat_mode="context", memory=memory)

        response = chat_engine.chat(user_query)

        st.markdown("### 📥 Response")
        st.info(response.response)

        # Save chat history
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append((user_query, response.response))

        st.markdown("### 🕒 Chat History")
        for i, (q, a) in enumerate(st.session_state['history'], 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("🔗 Want to embed this in your website or create a custom version for your business? [Contact Us](https://www.instrovate.com/contact) or [WhatsApp Us](https://wa.me/917428952788).")
