import streamlit as st
import datetime
import os
import sqlite3
from dotenv import load_dotenv
import traceback

# ---- SQLite Setup ----
DB_PATH = "chatgpt.sqlite"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Table for chat threads
    cur.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Table for messages
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER,
            role TEXT,
            content TEXT,
            time TEXT,
            FOREIGN KEY(thread_id) REFERENCES threads(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---- Helper functions ----
def list_threads():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM threads ORDER BY created DESC")
    threads = cur.fetchall()
    conn.close()
    return threads

def create_thread(name):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO threads (name) VALUES (?)", (name,))
    thread_id = cur.lastrowid
    conn.commit()
    conn.close()
    return thread_id

def rename_thread(thread_id, new_name):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE threads SET name=? WHERE id=?", (new_name, thread_id))
    conn.commit()
    conn.close()

def delete_thread(thread_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
    cur.execute("DELETE FROM threads WHERE id=?", (thread_id,))
    conn.commit()
    conn.close()

def save_message(thread_id, role, content, time):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (thread_id, role, content, time) VALUES (?, ?, ?, ?)",
        (thread_id, role, content, time)
    )
    conn.commit()
    conn.close()

def get_messages(thread_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content, time FROM messages WHERE thread_id=? ORDER BY id",
        (thread_id,)
    )
    msgs = cur.fetchall()
    conn.close()
    return [{"role": r, "content": c, "time": t} for r, c, t in msgs]

def clear_thread(thread_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
    conn.commit()
    conn.close()

# ---- Azure OpenAI Setup ----
load_dotenv()
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2025-01-01-preview"
)

# ---- Streamlit App ----
st.set_page_config(page_title="ChatGPT-like Azure Chat", page_icon="💬", layout="wide")

# --- Sidebar: Thread/Chat selection
st.sidebar.title("💬 Chats")

threads = list_threads()
if 'current_thread_id' not in st.session_state:
    # If no threads, create a default one
    if not threads:
        tid = create_thread("New Chat")
        st.session_state.current_thread_id = tid
    else:
        st.session_state.current_thread_id = threads[0][0]

# Select thread
thread_names = [name for _, name in threads]
thread_ids = [tid for tid, _ in threads]
selected_idx = 0
if st.session_state.current_thread_id in thread_ids:
    selected_idx = thread_ids.index(st.session_state.current_thread_id)
selected = st.sidebar.radio(
    "Conversations",
    options=thread_ids,
    format_func=lambda tid: [name for i, name in threads if i == tid][0],
    index=selected_idx
)
st.session_state.current_thread_id = selected

# --- Sidebar controls
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("➕", help="New Chat"):
        new_id = create_thread("New Chat")
        st.session_state.current_thread_id = new_id
        st.experimental_rerun()
with col2:
    if st.button("✏️", help="Rename"):
        new_name = st.text_input("Rename Chat", value=[name for i, name in threads if i == st.session_state.current_thread_id][0], key="renamebox")
        if st.button("Save", key="saverename"):
            rename_thread(st.session_state.current_thread_id, new_name)
            st.experimental_rerun()
with col3:
    if st.button("🗑️", help="Delete"):
        delete_thread(st.session_state.current_thread_id)
        threads = list_threads()
        if threads:
            st.session_state.current_thread_id = threads[0][0]
        else:
            st.session_state.current_thread_id = create_thread("New Chat")
        st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.write("Click 'New Chat' to start a new conversation.")

# --- Main Chat area
st.title("💬 ChatGPT-like Azure Chat")

messages = get_messages(st.session_state.current_thread_id)

for msg in messages:
    bubble = "bubble-user" if msg["role"] == "user" else "bubble-assistant"
    st.markdown(
        f"""
        <div style="background-color: {'#0078fe' if msg['role']=='user' else '#f4f4f8'};
                    color: {'white' if msg['role']=='user' else '#222'};
                    border-radius:1em; padding:12px 18px; margin-bottom:8px; max-width:70%; float:{'right' if msg['role']=='user' else 'left'}; box-shadow:0 2px 8px rgba(0,0,0,0.04); clear:both;">
        {msg['content']}
        <div style="font-size:0.75em;color:#888;margin-top:2px;">{msg['time']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Input area
with st.form("chat-form", clear_on_submit=True):
    user_input = st.text_area(
        "Type your message...",
        key="user_input",
        label_visibility="collapsed",
        height=80,
        max_chars=500
    )
    send = st.form_submit_button("Send", use_container_width=True)

if st.button("🧹 Clear Chat History", key="clearchat"):
    clear_thread(st.session_state.current_thread_id)
    st.experimental_rerun()

if send and user_input.strip():
    now = datetime.datetime.now().strftime("%H:%M")
    save_message(st.session_state.current_thread_id, "user", user_input, now)
    st.experimental_rerun()
    # After rerun, below code sends to Azure OpenAI and saves response

# Get latest messages after any new user message
messages = get_messages(st.session_state.current_thread_id)

# If last message is user and not yet answered by assistant, call API
if messages and messages[-1]['role'] == 'user' and (len(messages) == 1 or messages[-2]['role'] == 'assistant' or len(messages)==1):
    with st.spinner("Assistant is typing..."):
        try:
            ai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": m["role"], "content": m["content"]} for m in messages
                ],
                max_tokens=500
            )
            response_content = ai_response.choices[0].message.content
            now2 = datetime.datetime.now().strftime("%H:%M")
            save_message(st.session_state.current_thread_id, "assistant", response_content, now2)
            st.experimental_rerun()
        except Exception as e:
            st.error("Azure OpenAI API call failed!\n\n" + str(e))

# --- Tiny JS for auto-scroll
st.markdown("""
    <script>
    var chatDiv = window.parent.document.querySelector('section.main');
    if(chatDiv){ chatDiv.scrollTop = chatDiv.scrollHeight; }
    </script>
""", unsafe_allow_html=True)
