## 1. OpenAI way

## https://medium.com/streamlit/langchain-tutorial-1-build-an-llm-powered-app-in-18-lines-of-code-384a87016d0a

# import streamlit as st
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


## 2. Hugchat way


import streamlit as st
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


## 3. Transformer Way

## https://thepythoncode.com/article/conversational-ai-chatbot-with-huggingface-transformers-in-python


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


# for step in range(5):
#     # take user input
#     text = input(">> You:")
#     # encode the input and add end of string token
#     input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
#     # concatenate new user input with chat history (if there is)
#     bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids
#     # generate a bot response
#     chat_history_ids = model.generate(
#         bot_input_ids,
#         max_length=1000,
#         do_sample=True,
#         top_k=100,
#         temperature=0.75,
#         pad_token_id=tokenizer.eos_token_id
#     )
#     #print the output
#     output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
#     print(f"DialoGPT: {output}")


# DL: BERT + pytorch/tensorflow + huggingface
# train / fine-tune own transformer

# figure out what can install locally - transformers?
# fine-tune on kaggle and download to local?


## requirements backup

# openai
# langchain

# torch
# transformers
