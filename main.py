import os
import streamlit as st
from streamlit_chat import message as st_message
from dotenv import load_dotenv

from chat_utils import create_and_save_directory_embeddings
from streamlit_app_utils import process_summarize_button, generate_answer, load_db_from_file_and_create_if_not_exists, validate_api_key, load_dir_chat_embeddings

from summary_utils import transcript_loader

import pandas as pd

import glob




#Youtube stuff is kinda broken! I'll fix it soon.

load_dotenv('test.env')

st.set_page_config(page_title='BriefGPT')

accepted_filetypes = ['.txt', '.pdf', '.epub']

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
        files = [file for file in files if file.endswith(tuple(accepted_filetypes))]
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
    dir_or_doc = st.radio('Select a chat method', ('Document', 'Directory'))
    st.title('Chat')
    model_name = st.radio('Select a model', ('gpt-3.5-turbo', 'gpt-4'))
    hypothetical = st.checkbox('Use hypothetical embeddings', value=False)
    if dir_or_doc == 'Document':
        if 'text_input' not in st.session_state:
            st.session_state.text_input = ''
        directory = 'documents'
        files = os.listdir(directory)
        files = [file for file in files if file.endswith(tuple(accepted_filetypes))]
        selected_file = st.selectbox('Select a file', files)
        st.write('You selected: ' + selected_file)
        selected_file_path = os.path.join(directory, selected_file)

        if st.button('Load file (first time might take a second...) pressing this button will reset the chat history'):
            db = load_db_from_file_and_create_if_not_exists(selected_file_path)
            st.session_state.db = db
            st.session_state.history = []

    else:
        if 'text_input' not in st.session_state:
            st.session_state.text_input = ''
        load_or_create = st.checkbox('Load from existing directory (already embedded)', value=False)
        if load_or_create:
            embeddings = os.listdir('directory_embeddings')
            embeddings = [file for file in embeddings if file.endswith('.faiss')]
            select_embedding = st.selectbox('Select an embedding', embeddings)
            load = st.button('Load embeddings')
            if load:
                embedding_file_path = os.path.join('directory_embeddings', select_embedding)
                db = load_dir_chat_embeddings(embedding_file_path)
                st.session_state.db = db
                st.session_state.history = []

        else:
            directory = st.text_input('Enter a directory to load from - just "documents" will load the default documents folder')
            name = st.text_input('Enter a unique nickname for the directory')
            if st.button('Load directory (first time might take a second...) pressing this button will reset the chat history'):
                with st.spinner('Loading directory...'):
                    db = create_and_save_directory_embeddings(directory, name)
                    st.session_state.db = db
                    st.success('Directory loaded successfully')
                    st.session_state.history = []

    user_input = st.text_input('Enter your question', key='text_input')

    if st.button('Ask') and 'db' in st.session_state and validate_api_key(model_name):
        answer = generate_answer(st.session_state.db, model_name, hypothetical)

    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'sources' not in st.session_state:
        st.session_state.sources = []
    for i, chat in enumerate(st.session_state.history):
        st_message(**chat, key=str(i))
    for i, source in enumerate(st.session_state.sources):
        with st.expander('Sources', expanded=False):
            st.markdown(source)


def documents():
    st.title('Documents')
    st.markdown('Documents are stored in the documents folder in the project directory.')
    directory = 'documents'
    files = os.listdir(directory)
    files = [file for file in files if file.endswith(tuple(accepted_filetypes))]
    if files:
        files_df = pd.DataFrame(files, columns=['File Name'], index=range(1, len(files) + 1))
        st.dataframe(files_df, width=1000)
    else:
        st.write('No documents found in documents folder. Add some documents first!')


def compare_results():
    st.title('Compare')
    st.write("Compare retrieval results using hypothetical embeddings vs. normal embeddings. Support for comparing multiple models coming soon.")
    model_name = 'gpt-3.5-turbo'

    if 'text_input' not in st.session_state:
        st.session_state.text_input = ''
    directory = 'documents'
    files = os.listdir(directory)
    files = [file for file in files if file.endswith(tuple(accepted_filetypes))]
    selected_file = st.selectbox('Select a file', files)
    st.write('You selected: ' + selected_file)
    selected_file_path = os.path.join(directory, selected_file)

    if st.button('Load file (first time might take a second...) pressing this button will reset the chat history'):
        db = load_db_from_file_and_create_if_not_exists(selected_file_path)
        st.session_state.db = db
        st.session_state.history = []




    user_input = st.text_input('Enter your question', key='text_input')

    if st.button('Ask') and 'db' in st.session_state and validate_api_key(model_name):
        st.markdown('Question: ' + user_input)
        answer_a, sources_a = generate_answer(st.session_state.db, model_name, hypothetical=True)
        answer_b, sources_b = generate_answer(st.session_state.db, model_name, hypothetical=False)

        col1, col2 = st.columns(2)

        with col1:
            st.header('Hypothetical embeddings')
            st.markdown(answer_a)
            with st.expander('Sources', expanded=False):
                st.markdown(sources_a)
        with col2:
            st.header('Normal embeddings')
            st.markdown(answer_b)
            with st.expander('Sources', expanded=False):
                st.markdown(sources_b)

        st.session_state.history = []
        st.session_state.sources = []




PAGES = {
    "Chat": chat,
    "Summarize": summarize,
    "Documents": documents,
    "Compare": compare_results
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
st.sidebar.markdown(' [Contact author](mailto:ethanujohnston@gmail.com)')
st.sidebar.markdown(' [Github](https://github.com/e-johnstonn/docGPT)')
st.sidebar.markdown('[More info on hypothetical embeddings here](https://arxiv.org/abs/2212.10496)', unsafe_allow_html=True)
page = PAGES[selection]
page()






