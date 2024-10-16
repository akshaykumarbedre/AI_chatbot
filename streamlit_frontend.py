import streamlit as st
from vector_store import FlowerShopVectorStore
from chatbot import app, CustomState, get_user_history, clear_user_history
from langchain_core.messages import AIMessage, HumanMessage
from tools import customers_database, data_protection_checks

st.set_page_config(layout='wide', page_title='Flower Shop Chatbot', page_icon='💐')

# Initialize session state
if 'message_history' not in st.session_state:
    st.session_state.message_history = [AIMessage(content="Hiya, I'm the flower shop chatbot. How can I help?")]

if 'user_id' not in st.session_state:
    st.session_state.user_id = None

left_col, main_col, right_col = st.columns([1, 2, 1])

# 1. User ID input and buttons for chat
with left_col:
    user_id = st.text_input("Enter User ID:", key="user_id_input")
    if st.button('Set User ID'):
        st.session_state.user_id = user_id
        # Retrieve chat history for this user
        user_history = get_user_history(user_id)
        if user_history:
            st.session_state.message_history = user_history
        else:
            st.session_state.message_history = [AIMessage(content="Hiya, I'm the flower shop chatbot. How can I help?")]
        st.experimental_rerun()

    if st.button('Clear Chat'):
        st.session_state.message_history = [AIMessage(content="Hiya, I'm the flower shop chatbot. How can I help?")]
        # Clear chat history in the database
        if st.session_state.user_id:
            clear_user_history(st.session_state.user_id)
        st.experimental_rerun()

# 2. Chat history and input
with main_col:
    if st.session_state.user_id:
        st.write(f"Current User: {st.session_state.user_id}")
        user_input = st.chat_input("Type here...")
        if user_input:
            st.session_state.message_history.append(HumanMessage(content=user_input))
            
            state = CustomState(messages=st.session_state.message_history, user_id=st.session_state.user_id)
            response = app.invoke(state)
            
            st.session_state.message_history = response['messages']

        for i in range(len(st.session_state.message_history)):
            this_message = st.session_state.message_history[i]
            if isinstance(this_message, AIMessage):
                message_box = st.chat_message('assistant')
            else:
                message_box = st.chat_message('user')
            message_box.markdown(this_message.content)
    else:
        st.write("Please set a User ID to start chatting.")

# 3. State variables
with right_col:
    st.title('Customers Database')
    st.write(customers_database)
    st.title('Data Protection Checks')
    st.write(data_protection_checks)