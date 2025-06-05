import streamlit as st
import datetime
import os
import traceback
import numpy as np
from dotenv import load_dotenv

# ---- Azure OpenAI setup ----
try:
    load_dotenv()
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2025-01-01-preview"
    )
except Exception:
    st.error("Azure OpenAI initialization failed!\n\n" + traceback.format_exc())
    st.stop()

# ---- ChromaDB setup (with robust dummy embedding) ----
try:
    import chromadb

    # Dummy embedding function class that satisfies ChromaDB's requirements
    class DummyEmbeddingFunction:
        name = "default"
        def __call__(self, texts):
            # returns a fixed-size, valid float32 vector for each input
            return [np.zeros(384, dtype=np.float32) for _ in texts]

    chroma_client = chromadb.PersistentClient(path="./chat_db")
    collection = chroma_client.get_or_create_collection(
        "chat_history",
        embedding_function=DummyEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"}
    )
except Exception:
    st.error("ChromaDB initialization failed!\n\n" + traceback.format_exc())
    st.stop()

# ---- Persistence Helpers ----
def load_history():
    try:
        docs = collection.get(include=['documents', 'metadatas'])
        messages = []
        if docs.get('documents'):
            for doc, meta in zip(docs['documents'], docs['metadatas']):
                messages.append({
                    "role": meta.get('role', ''),
                    "content": doc,
                    "time": meta.get("time", "")
                })
        return messages
    except Exception:
        st.error("Failed to load chat history!\n\n" + traceback.format_exc())
        return []

def save_history(messages):
    try:
        ids = [str(i) for i in range(len(messages))]
        roles = [msg["role"] for msg in messages]
        contents = [msg["content"] for msg in messages]
        times = [msg.get("time", "") for msg in messages]
        old_ids = collection.get()['ids']
        if old_ids:
            collection.delete(ids=old_ids)
        if contents:
            collection.add(
                documents=contents,
                metadatas=[{"role": r, "time": t} for r, t in zip(roles, times)],
                ids=ids
            )
    except Exception:
        st.error("Failed to save chat history!\n\n" + traceback.format_exc())

def clear_history():
    try:
        old_ids = collection.get()['ids']
        if old_ids:
            collection.delete(ids=old_ids)
    except Exception:
        st.error("Failed to clear chat history!\n\n" + traceback.format_exc())

# ---- Streamlit Page Setup ----
st.set_page_config(page_title="ChatGPT-like Azure Chat", page_icon="💬", layout="wide")
st.markdown("<style>textarea{font-size:1.1em;}</style>", unsafe_allow_html=True)
st.title("💬 ChatGPT-like Azure Chat")

# ---- Load messages into session_state ----
if 'messages' not in st.session_state:
    try:
        st.session_state.messages = load_history()
    except Exception:
        st.session_state.messages = []
        st.error("Failed to load messages at startup!\n\n" + traceback.format_exc())

# ---- Styling ----
st.markdown("""
    <style>
    .bubble-user {
        background: linear-gradient(135deg, #0078fe 0%, #19c2fa 100%);
        color: white;
        border-radius: 1em 1em 0.1em 1em;
        padding: 12px 18px;
        margin: 8px 0 8px 20%;
        max-width: 80%;
        float: right;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .bubble-assistant {
        background: #f4f4f8;
        color: #222;
        border-radius: 1em 1em 1em 0.1em;
        padding: 12px 18px;
        margin: 8px 20% 8px 0;
        max-width: 80%;
        float: left;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .bubble-time {
        font-size:0.75em;color:#888;margin-top:2px;
    }
    .chat-row:after { content: ""; display: table; clear: both; }
    </style>
""", unsafe_allow_html=True)

# ---- Chat Display ----
try:
    for msg in st.session_state.messages:
        bubble = "bubble-user" if msg["role"] == "user" else "bubble-assistant"
        time_str = msg.get("time", "")
        st.markdown(
            f"<div class='chat-row'><div class='{bubble}'>{msg['content']}<div class='bubble-time'>{time_str}</div></div></div>",
            unsafe_allow_html=True
        )
except Exception:
    st.error("Error displaying chat messages!\n\n" + traceback.format_exc())

# ---- Input Form ----
with st.form("chat-form", clear_on_submit=True):
    user_input = st.text_area(
        "Type your message...",
        key="user_input",
        label_visibility="collapsed",
        height=80,  # <-- You requested 80px height
        max_chars=500
    )
    send = st.form_submit_button("Send", use_container_width=True)

# ---- Clear Chat Button ----
col1, col2 = st.columns([9,1])
with col2:
    if st.button("🗑️", help="Clear Chat"):
        try:
            clear_history()
            st.session_state.messages = []
            st.rerun()
        except Exception:
            st.error("Failed to clear chat!\n\n" + traceback.format_exc())

# ---- Message Sending Logic ----
if send and user_input.strip():
    try:
        now = datetime.datetime.now().strftime("%H:%M")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "time": now
        })
        save_history(st.session_state.messages)
        with st.spinner("Assistant is typing..."):
            try:
                ai_response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Your deployment/model name
                    messages=[
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages
                    ],
                    max_tokens=500
                )
                response_content = ai_response.choices[0].message.content
                now2 = datetime.datetime.now().strftime("%H:%M")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_content,
                    "time": now2
                })
                save_history(st.session_state.messages)
                st.rerun()
            except Exception:
                st.error("Azure OpenAI API call failed!\n\n" + traceback.format_exc())
    except Exception:
        st.error("Failed to send or save your message!\n\n" + traceback.format_exc())

# ---- Auto-scroll JS ----
st.markdown("""
    <script>
    var chatDiv = window.parent.document.querySelector('section.main');
    if(chatDiv){ chatDiv.scrollTop = chatDiv.scrollHeight; }
    </script>
""", unsafe_allow_html=True)
