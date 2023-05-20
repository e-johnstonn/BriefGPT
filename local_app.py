import os

import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from streamlit_chat import message as st_message

import pandas as pd

from local_chat_utils import load_db_from_file_and_create_if_not_exists_local
from streamlit_app_utils import generate_answer

from langchain.llms import GPT4All, LlamaCpp

from dotenv import load_dotenv



load_dotenv('test.env')

model_type = os.getenv('MODEL_TYPE')
model_path = os.getenv('MODEL_PATH')
docs_lang = os.getenv('DOCS_LANG')

accepted_filetypes = ['.txt', '.pdf', '.epub']

#Model is initialized here. Configure it with your parameters and the path to your model.

loading = st.spinner('Initializing LLM')
with st.spinner('Initializing LLM...'):
    if 'llm' not in st.session_state:
        with st.spinner('Loading LLM...'):
            if model_type.upper() == 'LlamaCpp'.upper():
                llm = LlamaCpp(model_path=model_path, n_ctx=1000)
                st.session_state.llm = llm
            elif model_type.upper() == 'GPT4All'.upper():
                llm = GPT4All(model=model_path, backend='gptj', n_ctx=1000)
                st.session_state.llm = llm
            else:
                st.warning('Invalid model type. GPT4ALL or LlamaCpp supported - make sure you specify in your env file.')


def chat():
    st.title('Chat')
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ''
    directory = 'documents'
    files = os.listdir(directory)
    files = [file for file in files if file.endswith(tuple(accepted_filetypes))]
    selected_file = st.selectbox('Select a file', files)
    st.write('You selected: ' + selected_file)
    selected_file_path = os.path.join(directory, selected_file)

    if st.button('Load file (first time might take a second...) pressing this button will reset the chat history'):
        db = load_db_from_file_and_create_if_not_exists_local(selected_file_path, docs_lang)
        st.session_state.db = db
        st.session_state.history = []

    user_input = st.text_input('Enter your question', key='text_input')

    if st.button('Ask') and 'db' in st.session_state:
        answer = generate_answer(st.session_state.db, st.session_state.llm)


    if 'history' not in st.session_state:
        st.session_state.history = []
    for i, chat in enumerate(st.session_state.history):
        st_message(**chat, key=str(i))




def documents():
    st.title('Documents')
    st.markdown('Documents are stored in the documents folder in the project directory.')
    directory = 'documents'
    files = os.listdir(directory)
    files = [file for file in files if file.endswith('.txt') or file.endswith('.pdf')]
    if files:
        files_df = pd.DataFrame(files, columns=['File Name'], index=range(1, len(files) + 1))
        st.dataframe(files_df, width=1000)
    else:
        st.write('No documents found in documents folder. Add some documents first!')



PAGES = {
    "Chat": chat,
    "Documents": documents,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
st.sidebar.markdown(' [Contact author](mailto:ethanujohnston@gmail.com)')
st.sidebar.markdown(' [Github](https://github.com/e-johnstonn/docsummarizer)')
page = PAGES[selection]
page()