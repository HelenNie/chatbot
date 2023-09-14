# https://medium.com/streamlit/langchain-tutorial-1-build-an-llm-powered-app-in-18-lines-of-code-384a87016d0a


import streamlit as st
# from langchain.llms import OpenAI

# st.set_page_config(page_title="🔗 Helen's Chatbot")

# openai_api_key = st.sidebar.text_input('OpenAI API Key')

# def generate_response(input_text):
#   llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
#   st.info(llm(input_text))

# with st.form('my_form'):
#   text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#   submitted = st.form_submit_button('Submit')
#   if not openai_api_key.startswith('sk-'):
#     st.warning('Please enter your OpenAI API key!', icon='⚠')
#   if submitted and openai_api_key.startswith('sk-'):
#     generate_response(text)


from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat
from hugchat.login import Login


st.set_page_config(page_title="🔗 Helen's Chatbot")


# Log in to huggingface and grant authorization to huggingchat
email = st.secrets["HC_USERNAME"]
passwd = st.secrets["HC_PASSWD"]
sign = Login(email, passwd)
cookies = sign.login()

# Save cookies to the local directory
cookie_path_dir = "./cookies_snapshot"
sign.saveCookiesToDir(cookie_path_dir)


input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()


if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm ChaiTea, How may I help you?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# User input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

# Response output
def generate_response(prompt):
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    response = chatbot.chat(prompt)
    return response

with st.sidebar:
    st.title('💬 ChaiTea the Chatbot')
    st.markdown('''
    	Hello!
    ''')
    add_vertical_space(5)
    st.write('By Helen Nie')

## Applying the user input box
with input_container:
    user_input = get_text()

with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['generated'][i], key=str(i))

# deep learning alternative: BERT + pytorch ?



