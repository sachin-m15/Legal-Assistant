import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# --- Import the router instead of the agent app directly ---
from agent.router import route_query

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Legal Assistant", layout="wide")


# ====================== Utility Functions ======================
def generate_thread_id():
    return str(uuid.uuid4())


def initial_greeting():
    if not st.session_state.messages:
        if st.session_state["role"] == "Lawyer":
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Hello, Lawyer! Please provide the case details you are working on.",
                }
            )
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Hello, how can I assist you with your legal query today?",
                }
            )


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])
    st.session_state["messages"] = []  # Reset Streamlit's native message history
    # Reset the initial greeting based on the current role
    initial_greeting()


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# ====================== Session State ======================
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [st.session_state["thread_id"]]

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "messages" not in st.session_state:  # ✅ FIX: initialize messages
    st.session_state["messages"] = []

if "role" not in st.session_state:  # Default role
    st.session_state["role"] = "Common Citizen"


# ====================== UI Layout ======================
st.sidebar.title("⚖️ L.A.R.A (Legal Analysis & Research Assistant)")
if st.sidebar.button("➕ New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.subheader("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    title = f"Chat {thread_id[:6]}"
    if st.sidebar.button(title, key=thread_id):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = []
        st.rerun()

st.title("💬 LARA – Your AI Legal Companion")

# --- Role Selection UI ---
selected_role = st.radio(
    "Select your role:", ("Common Citizen", "Lawyer"), key="role_selector"
)
if selected_role != st.session_state["role"]:
    st.session_state["role"] = selected_role
    reset_chat()
    st.rerun()

# Ensure greeting exists
initial_greeting()

st.markdown(
    "Ask a question related to Indian law and I’ll provide research & analysis."
)


# ====================== Display History ======================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ====================== Chat Input ======================
user_input = st.chat_input("Type your legal query here...")

if user_input:
    # Add user's message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Route the query
    with st.chat_message("assistant"):
        try:
            result = route_query(
                role=st.session_state["role"],
                user_query=user_input,
                thread_id=st.session_state["thread_id"],
            )

            # Extract and display final analysis
            full_analysis = result.get("final_analysis", "⚠️ No response generated.")
            st.markdown(full_analysis)

            # Save AI response
            st.session_state.messages.append(
                {"role": "assistant", "content": full_analysis}
            )

        except Exception as e:
            st.error(f"Error: {e}")
