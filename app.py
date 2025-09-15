import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from agent import app, retrieve_all_threads  # noqa: F401

load_dotenv()
st.set_page_config(page_title="Legal Assistant", layout="wide")


# ====================== Utility Functions ======================
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation_history(thread_id):
    try:
        state = app.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.get("chat_history", [])
    except Exception as e:
        st.error(f"Could not load history for {thread_id}: {e}")
        return []


# ====================== Session State ======================
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [st.session_state["thread_id"]]

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# ====================== UI Layout ======================
st.sidebar.title("⚖️L.A.R.A (Legal Analysis & Research Assistant)")
if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.subheader("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    title = f"Chat {thread_id[:6]}"
    if st.sidebar.button(title):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation_history(thread_id)
        st.rerun()

st.title("💬LARA – Your AI Legal Companion")
st.markdown(
    "Ask a question related to Indian law and I’ll provide research & analysis."
)


# ====================== Display History ======================
for message in st.session_state["message_history"]:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)


# ====================== Chat Input ======================
user_input = st.chat_input("Type your legal query here...")

if user_input:
    # Add user's message
    st.session_state["message_history"].append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

            # ✅ Use app.invoke() instead of app.execute()
            result = app.invoke(
                {
                    "query": user_input,
                    "chat_history": st.session_state["message_history"],
                },
                config=CONFIG,
            )

            # Extract only the final_analysis content
            full_analysis = result.get("final_analysis", "")
            if isinstance(full_analysis, dict):
                full_analysis = full_analysis.get("content", "")
            full_analysis = str(full_analysis)

            # Display AI output
            st.markdown(full_analysis)

            # Add AI response to history
            st.session_state["message_history"].append(AIMessage(content=full_analysis))

        except Exception as e:
            st.error(f"Error: {e}")
