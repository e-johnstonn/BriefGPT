import os
import streamlit as st
from streamlit_chat import message as st_message
from dotenv import load_dotenv
from streamlit_app_utils import process_summarize_button, generate_answer, load_db_from_file_and_create_if_not_exists, validate_api_key

from summary_utils import transcript_loader

import pandas as pd

import glob




#Youtube stuff is kinda broken! I'll fix it soon.

load_dotenv('test.env')

st.set_page_config(page_title='BriefGPT')

def summarize():
    """
    The main function for the Streamlit app.

    :return: None.
    """
    st.title("Summarize")
    st.write("Summaries are saved to the 'summaries' folder in the project directory.")

    input_method = st.radio("Select input method", ('Document', 'YouTube URL'))

    if input_method == 'Document':
        directory = 'documents'
        files = os.listdir(directory)
        files = [file for file in files if file.endswith('.txt') or file.endswith('.pdf')]
        if files:
            selected_file = st.selectbox('Select a file', files)
            st.write('You selected: ' + selected_file)
            selected_file_path = os.path.join(directory, selected_file)
        else:
            st.write('No documents found in documents folder. Add some documents first!')
            return

    if input_method == 'YouTube URL':
        youtube_url = st.text_input("Enter a YouTube URL to summarize")

    use_gpt_4 = st.checkbox("Use GPT-4 for the final prompt (STRONGLY recommended, requires GPT-4 API access - progress bar will appear to get stuck as GPT-4 is slow)", value=True)
    find_clusters = st.checkbox('Optimal clustering (saves on tokens)', value=False)



    if st.button('Summarize (click once and wait)'):
        if input_method == 'Document':
            process_summarize_button(selected_file_path, use_gpt_4, find_clusters)

        else:
            doc = transcript_loader(youtube_url)
            process_summarize_button(doc, use_gpt_4, find_clusters, file=False)



def chat():
    st.title('Chat')
    model_name = st.radio('Select a model', ('gpt-3.5-turbo', 'gpt-4'))
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ''
    directory = 'documents'
    files = os.listdir(directory)
    files = [file for file in files if file.endswith('.txt') or file.endswith('.pdf')]
    selected_file = st.selectbox('Select a file', files)
    st.write('You selected: ' + selected_file)
    selected_file_path = os.path.join(directory, selected_file)

    if st.button('Load file (first time might take a second...) pressing this button will reset the chat history'):
        db = load_db_from_file_and_create_if_not_exists(selected_file_path)
        st.session_state.db = db
        st.session_state.history = []

    user_input = st.text_input('Enter your question', key='text_input')

    if st.button('Ask') and 'db' in st.session_state and validate_api_key(model_name):
        answer = generate_answer(st.session_state.db, model_name)


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
    "Summarize": summarize,
    "Documents": documents,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
st.sidebar.markdown(' [Contact author](mailto:ethanujohnston@gmail.com)')
st.sidebar.markdown(' [Github](https://github.com/e-johnstonn/docGPT)')
page = PAGES[selection]
page()






