import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
  load_dotenv()
# 1. Page Configuration
st.set_page_config(
    page_title="India's Got Latent — Contestant AI",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS for stage aesthetic
st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .stSelectbox label { color: #facc15 !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🎙️ India's Got Latent — AI Stage")
st.caption("The judges are watching. Select your act persona and start the banter.")

# 2. Sidebar API & Persona Setup
with st.sidebar:
    st.header("⚙️ Backstage Setup")
    default_key = os.getenv("GROQ_API_KEY", "")
  api_key = st.text_input(
      "Enter Groq API Key",
      type="password",
      value=default_key,
      help="Leave blank to use your own key; defaults to a local .env if set."
  )
    
    PERSONAS = {
        "Default (No Persona)": "You are a helpful, direct conversational AI assistant.",
        "RoastBot 💣": (
            "You are a contestant on 'India's Got Latent' doing a roast comedy act. "
            "Deliver sharp, witty, and hilarious roasts to the judges (like Samay Raina or guest judges). "
            "Keep responses under 3-4 sentences. Be edgy and funny, but avoid genuine hate speech."
        ),
        "ShakespeareBot 🎭": (
            "You are a 16th-century Elizabethan playwright performing on an Indian comedy stage. "
            "Speak entirely in dramatic, poetic Shakespearean English using 'thou', 'thee', 'doth', "
            "and dramatic theatrical flair."
        ),
        "Emoji Translator 🤪": (
            "You translate every user thought into a chaotic sequence of emojis, "
            "followed by 1 sentence explaining the vibe in slang."
        ),
        "Strict Hostel Warden 🧹": (
            "You are an overly strict Indian hostel warden. Interrogate the judges about their curfew, "
            "discipline, and threaten them with calling their parents."
        )
    }

    selected_persona = st.selectbox("Choose Your Stage Act:", list(PERSONAS.keys()))

    if st.button("Reset Stage / Clear History"):
        st.session_state.chat_history = InMemoryChatMessageHistory()
        st.session_state.messages = []
        st.rerun()

# Check for API Key
if not api_key:
    st.info("👈 Please enter your Groq API key in the sidebar to boot up the bot!")
    st.stop()

# 3. Memory & Session Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. LangChain Chain Setup
system_instructions = PERSONAS[selected_persona]

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instructions + "\n\nAlways maintain proper formatting with paragraphs or bullets when helpful."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

chain = prompt | llm

# Wrap chain with conversation memory wrapper
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda session_id: st.session_state.chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 5. Render Previous Chat History in UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Chat Input & Processing Loop
if user_input := st.chat_input("Say something to the bot..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Render bot response with streaming feel
    with st.chat_message("assistant"):
        with st.spinner("The bot is thinking..."):
            response = conversational_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "latentshow"}}
            )
            bot_reply = response.content
            st.markdown(bot_reply)

    # Save bot response to session display
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
