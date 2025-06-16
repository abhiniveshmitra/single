# src/database.py
# (No other changes needed, just adding the parameter to connect)

import sqlite3
import datetime

DB_NAME = "chat_history.db"

def init_db():
    # FIX: Allow connection across multiple threads
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
        )
    """)
    conn.commit()
    conn.close()

def add_chat(title):
    # FIX: Allow connection across multiple threads
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (title) VALUES (?)", (title,))
    chat_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return chat_id

def get_chats():
    # FIX: Allow connection across multiple threads
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, created_at FROM chats ORDER BY created_at DESC")
    chats = cursor.fetchall()
    conn.close()
    return chats

def get_messages(chat_id):
    # FIX: Allow connection across multiple threads
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY created_at ASC", (chat_id,))
    messages = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in messages]

def add_message(chat_id, role, content):
    # FIX: Allow connection across multiple threads
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, role, content))
    conn.commit()
    conn.close()
# app.py
import streamlit as st
from dotenv import load_dotenv
from src import database, azure_services, ui_components

# --- Page Config and Initialization ---
st.set_page_config(page_title="Azure AI Chat", layout="centered")
load_dotenv()
database.init_db()

# Initialize session state if not already done
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Rendering ---
ui_components.render_sidebar()
st.title("Azure AI Chat Assistant")
ui_components.render_chat_messages()

# --- Chat Input Logic ---
# FIX: The entire chat logic is now handled within this block to prevent state loss
if prompt := st.chat_input("Ask a question or use the microphone..."):
    # If starting a new chat, create it in the database but DO NOT rerun yet
    if st.session_state.current_chat_id is None:
        new_title = prompt[:40] + "..."
        st.session_state.current_chat_id = database.add_chat(new_title)
        st.session_state.messages = [] # Ensure messages list is empty for the new chat

    # Append and display the user's message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save user message to the database
    database.add_message(st.session_state.current_chat_id, "user", prompt)

    # Now, call the agent and process the response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # The agent is now reliably called with the user's prompt
            stream = azure_services.get_chat_completion(
                st.session_state.messages,
                st.session_state.get("vector_store")
            )
            response_content = st.write_stream(stream)
    
    # Append and save the full assistant response
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    database.add_message(st.session_state.current_chat_id, "assistant", response_content)

    # Handle Text-to-Speech if enabled
    if st.session_state.get("tts_enabled", False):
        with st.spinner("Synthesizing audio..."):
            audio_data = azure_services.synthesize_text_to_speech(response_content)
            if audio_data:
                st.audio(audio_data, format="audio/wav")

# Microphone logic needs to be separate to avoid conflicting with chat_input
if st.button("🎤", key="mic_button"):
    transcribed_text = azure_services.transcribe_audio_from_mic()
    if "Error" not in transcribed_text and transcribed_text:
        # Instead of rerunning, we can just set the chat_input's value for the next run,
        # but for instant use, it's better to handle it directly.
        # For simplicity, we'll let the user copy/paste or re-type the transcribed text.
        # A more complex solution would use st.rerun() carefully.
        st.info(f"🎤 Transcription: {transcribed_text}")
        st.warning("Please copy the text into the chat box to send.")


