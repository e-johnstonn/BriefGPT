import os
import streamlit as st
from streamlit_chat import message as st_message
from dotenv import load_dotenv

from chat_utils import create_and_save_directory_embeddings
from streamlit_app_utils import process_summarize_button, generate_answer, load_db_from_file_and_create_if_not_exists, validate_api_key, load_dir_chat_embeddings
from trubrics.integrations.streamlit import FeedbackCollector
from summary_utils import transcript_loader

import pandas as pd

import glob


email = st.secrets.get("TRUBRICS_EMAIL")
password = st.secrets.get("TRUBRICS_PASSWORD")

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
    
    if 'input' not in st.session_state:
        st.session_state.input = False
    
    if 'summary' not in st.session_state:
        st.session_state.summary = ''
    
    if 'name' not in st.session_state:
        st.session_state.name = ''
    
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''

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
        st.session_state.input = True
        if input_method == 'Document':
            st.session_state.model_name,st.session_state.summary,st.session_state.name = process_summarize_button(selected_file_path, use_gpt_4, find_clusters)

        else:
            doc = transcript_loader(youtube_url)
            st.session_state.model_name,st.session_state.summary,st.session_state.name = process_summarize_button(doc, use_gpt_4, find_clusters, file=False)

    if st.session_state.input:

        st.markdown(st.session_state.summary, unsafe_allow_html=True)
        with open(f'summaries/{st.session_state.name}_summary.txt', 'w') as f:
            f.write(st.session_state.summary)
        st.text(f' Summary saved to summaries/{st.session_state.name}_summary.txt')

        collector = FeedbackCollector(
            component_name="default",
            email=email,
            password=password,
        )
        
        feedback = collector.st_feedback(
            feedback_type="faces",
            model=st.session_state.model_name,
            open_feedback_label="[Optional] Provide additional feedback",
            metadata={
                "user_input": input_method,
                "summary": st.session_state.summary,
            },
            tags=["summary"],
        )







def chat():

    if 'answer' not in st.session_state:
        st.session_state.answer = ''
    
    if 'answered' not in st.session_state:
        st.session_state.answered = False
    
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''

    dir_or_doc = st.radio('Select a chat method', ('Document', 'Directory'))
    st.title('Chat')
    model_name = st.radio('Select a model', ('gpt-3.5-turbo', 'gpt-4'))
    st.session_state['model_name'] = model_name
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
        st.session_state['answered'] = True
        st.session_state['answer'] = answer 


    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'sources' not in st.session_state:
        st.session_state.sources = []
    for i, chat in enumerate(st.session_state.history):
        st_message(**chat, key=str(i))
    for i, source in enumerate(st.session_state.sources):
        with st.expander('Sources', expanded=False):
            st.markdown(source)

    if st.session_state['answered']:
        
        collector = FeedbackCollector(
            component_name="default",
            email=email,
            password=password,
        )

        feedback = collector.st_feedback(
            feedback_type="faces",
            model=st.session_state.model_name,
            open_feedback_label="[Optional] Provide additional feedback",
            metadata={
                "user_input": user_input,
                "answer": st.session_state['answer'][0],
            },
            tags=["chat"],
        )

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
    if 'input' not in st.session_state:
        st.session_state.input = False
    if 'answer_a' not in st.session_state:
        st.session_state.answer_a = ''
    if 'answer_b' not in st.session_state:
        st.session_state.answer_b = ''
    if 'sources_a' not in st.session_state:
        st.session_state.sources_a = ''
    if 'sources_b' not in st.session_state:
        st.session_state.sources_b = ''
    
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
        st.session_state.input=True
        st.markdown('Question: ' + user_input)
        st.session_state.answer_a, st.session_state.sources_a = generate_answer(st.session_state.db, model_name, hypothetical=True)
        st.session_state.answer_b, st.session_state.sources_b = generate_answer(st.session_state.db, model_name, hypothetical=False)


    if st.session_state.input:
        
        col1, col2 = st.columns(2)

        with col1:
            st.header('Hypothetical embeddings')
            st.markdown(st.session_state.answer_a)
            with st.expander('Sources', expanded=False):
                st.markdown(st.session_state.sources_a)
        with col2:
            st.header('Normal embeddings')
            st.markdown(st.session_state.answer_b)
            with st.expander('Sources', expanded=False):
                st.markdown(st.session_state.sources_b)

        st.session_state.history = []
        st.session_state.sources = []

        collector = FeedbackCollector(
            component_name="default",
            email=email,
            password=password,
        )

        feedback = collector.st_feedback(
            feedback_type="textbox",
            model=model_name,
            open_feedback_label="[Optional] Provide additional feedback",
            metadata={
                "user_input": user_input,
                "answer_a": st.session_state.answer_a,
                "answer_b": st.session_state.answer_b,
                "sources_a": st.session_state.sources_a,
                "sources_b": st.session_state.sources_b,
            },
            tags=["compare"],
        )




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






