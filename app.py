import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# --- Import the router instead of the agent app directly ---
from agent.router import route_query

load_dotenv()
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
    st.session_state.messages = []  # Reset Streamlit's native message history
    # Reset the initial greeting based on the current role
    initial_greeting()


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# The load_conversation_history needs to be updated to work with the router
# The LangGraph MemorySaver does not expose a list of threads for retrieval
# in a simple manner, so we'll handle history directly in the session state for this example.

# ====================== Session State ======================
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [st.session_state["thread_id"]]

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# --- NEW: Session State for Role ---
if "role" not in st.session_state:
    st.session_state["role"] = "Common Citizen"

# ====================== UI Layout ======================
st.sidebar.title("⚖️L.A.R.A (Legal Analysis & Research Assistant)")
if st.sidebar.button("➕ New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.subheader("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    title = f"Chat {thread_id[:6]}"
    if st.sidebar.button(title, key=thread_id):
        st.session_state["thread_id"] = thread_id
        # Reloading history requires a database or more complex retrieval,
        # so for this demo, we'll just reset the chat to a new thread.
        # For a full implementation, you'd load from the checkpointer.
        st.session_state["message_history"] = []
        st.rerun()

st.title("💬LARA – Your AI Legal Companion")

# --- NEW: Role Selection UI ---
selected_role = st.radio(
    "Select your role:", ("Common Citizen", "Lawyer"), key="role_selector"
)
# Update the session state with the selected role
if selected_role != st.session_state["role"]:
    st.session_state["role"] = selected_role
    # Reset the chat to provide a new, role-specific greeting
    reset_chat()
    st.rerun()

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
    # Add user's message to the session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- NEW: Call the router function ---
    with st.chat_message("assistant"):
        try:
            # Pass the user's role to the router
            result = route_query(
                role=st.session_state["role"],
                user_query=user_input,
                thread_id=st.session_state["thread_id"],
            )

            # Extract the final_analysis content from the result
            full_analysis = result.get("final_analysis", "")

            # Display AI output
            st.markdown(full_analysis)

            # Add AI response to history
            st.session_state.messages.append(
                {"role": "assistant", "content": full_analysis}
            )

        except Exception as e:
            st.error(f"Error: {e}")
