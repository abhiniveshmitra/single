import streamlit as st
from dotenv import load_dotenv
import os
from ingestion import (
    save_uploaded_file,
    save_uploaded_image,
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text_from_image,
    is_image,
)
from embedding_retrieval import chunk_text, build_embedding_index, get_top_chunks
from PIL import Image
from openai import AzureOpenAI
import uuid

load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
GPT_DEPLOYMENT = os.getenv("GPT_MODEL", "gpt-4-1106-preview")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

st.set_page_config(page_title="ChatGPT+ Semantic File Q&A", layout="wide")

# Chat history: each item is dict with keys: 'role', 'type', 'content'
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "doc_chunks" not in st.session_state:
    st.session_state["doc_chunks"] = []
if "doc_embeddings" not in st.session_state:
    st.session_state["doc_embeddings"] = None

# CSS for center-aligned, modern chat bubbles and area
st.markdown("""
<style>
.main { background-color: #f9f9f9; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.chat-container {
    max-width: 660px;
    margin: 0 auto;
    padding: 30px 0 100px 0;
}
.bubble-user {
    background: #cce7ff; border-radius:20px; padding:10px 18px; margin:8px 0;
    max-width:72%; float: right; clear: both; text-align: right; font-size: 1.1em;
}
.bubble-assistant {
    background: #eee; border-radius:20px; padding:10px 18px; margin:8px 0;
    max-width:72%; float: left; clear: both; text-align: left; font-size: 1.1em;
}
.bubble-img {
    border-radius: 14px; border: 2px solid #cce7ff; margin:10px 0; float: right; max-width:200px;
    display: block;
}
.bubble-system {
    background: #fff9c4; border-radius:14px; padding:9px 16px; margin:6px 0; font-size:0.99em; max-width:65%; float: left; clear: both;
}
.msg-clear { clear: both; }
.stTextInput > div > div > input {
    font-size: 1.12em;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown('''<h2 style="text-align:center; font-size:2.1em; margin-bottom: 10px;">💬 ChatGPT+ Semantic File Q&A</h2>
<p style="text-align:center; margin-bottom: 30px;">
Ask a question, or upload/paste a file/image using the 📎 icon or "Paste image" button below.
</p>
''', unsafe_allow_html=True)

# --- CHAT BUBBLES AREA ---
for entry in st.session_state.chat_history:
    role = entry["role"]
    if entry["type"] == "text":
        if role == "user":
            st.markdown(f'<div class="bubble-user">{entry["content"]}</div><div class="msg-clear"></div>', unsafe_allow_html=True)
        elif role == "assistant":
            st.markdown(f'<div class="bubble-assistant">{entry["content"]}</div><div class="msg-clear"></div>', unsafe_allow_html=True)
        elif role == "system":
            st.markdown(f'<div class="bubble-system">{entry["content"]}</div><div class="msg-clear"></div>', unsafe_allow_html=True)
    elif entry["type"] == "image":
        st.markdown(f'<img src="data:image/png;base64,{entry["content"]}" class="bubble-img"><div class="msg-clear"></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- CHAT INPUT AREA: Text input + Upload + Paste image ---
c1, c2, c3 = st.columns([7,1,3])
with c1:
    user_prompt = st.text_input("Type your message...", key=str(uuid.uuid4()), label_visibility="collapsed")
with c2:
    uploaded_file = st.file_uploader("📎", type=["txt", "pdf", "jpg", "jpeg", "png"], label_visibility="collapsed", key=str(uuid.uuid4()))
with c3:
    pasted_img = st.camera_input("Paste or capture image", key=str(uuid.uuid4()))

# --- Handle File Uploads (as chat message) ---
if uploaded_file is not None:
    filename = uploaded_file.name
    ext = filename.lower().split(".")[-1]
    if is_image(filename):
        file_path = save_uploaded_image(uploaded_file, filename)
        import base64
        with open(file_path, "rb") as imgf:
            b64 = base64.b64encode(imgf.read()).decode()
        st.session_state.chat_history.append({"role":"user","type":"image","content":b64})
        ocr_text = extract_text_from_image(file_path)
        # Add as context (not shown)
        st.session_state.doc_chunks.append(ocr_text)
    elif ext == "txt":
        file_path = save_uploaded_file(uploaded_file)
        txt = extract_text_from_txt(file_path)
        st.session_state.chat_history.append({"role":"user","type":"text","content":f"[Uploaded TXT: {filename}]"})
        for chunk in chunk_text(txt):
            st.session_state.doc_chunks.append(chunk)
    elif ext == "pdf":
        file_path = save_uploaded_file(uploaded_file)
        txt, ocr_msgs = extract_text_from_pdf(file_path)
        st.session_state.chat_history.append({"role":"user","type":"text","content":f"[Uploaded PDF: {filename}]"})
        for chunk in chunk_text(txt):
            st.session_state.doc_chunks.append(chunk)
    else:
        st.session_state.chat_history.append({"role":"assistant","type":"text","content":f"Unsupported file type: {filename}"})

# --- Handle Pasted Images (as chat message) ---
if pasted_img is not None:
    img = Image.open(pasted_img)
    temp_img_path = f"data/uploads/pasted_{uuid.uuid4().hex}.png"
    img.save(temp_img_path)
    import base64
    with open(temp_img_path, "rb") as imgf:
        b64 = base64.b64encode(imgf.read()).decode()
    st.session_state.chat_history.append({"role":"user","type":"image","content":b64})
    ocr_text = extract_text_from_image(temp_img_path)
    st.session_state.doc_chunks.append(ocr_text)

# --- Build/update embedding index ---
if st.session_state.doc_chunks:
    st.session_state.doc_embeddings = build_embedding_index(st.session_state.doc_chunks)

# --- Handle text input (send) ---
if st.button("Send", use_container_width=True) and user_prompt.strip():
    st.session_state.chat_history.append({"role":"user","type":"text","content":user_prompt})
    retrieved_chunks = []
    if st.session_state.doc_embeddings is not None and st.session_state.doc_chunks:
        retrieved_chunks = get_top_chunks(user_prompt, st.session_state.doc_chunks, st.session_state.doc_embeddings, top_k=4)
    messages = [{"role":"system","content":"You are a helpful assistant. If there is context, use it to answer the user's question."}]
    if retrieved_chunks:
        context_text = "\n\n".join(retrieved_chunks)
        messages.append({"role":"system","content":f"Context:\n{context_text[:3000]}"})
    messages.append({"role":"user","content":user_prompt})
    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            messages=messages,
            max_tokens=512,
        )
        answer = response.choices[0].message.content
    st.session_state.chat_history.append({"role":"assistant","type":"text","content":answer})

# --- Auto scroll to bottom JS hack ---
st.markdown("""
    <script>
        var body = window.parent.document.querySelector('.main');
        body.scrollTop = body.scrollHeight;
    </script>
""", unsafe_allow_html=True)
