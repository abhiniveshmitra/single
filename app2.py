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
        height=80,  # <-- 80px height as requested
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
                    model="gpt-4o-mini",  # Your model name
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
