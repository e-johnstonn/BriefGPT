import PyPDF2

from io import StringIO

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

from chat_utils import load_chat_embeddings, create_and_save_chat_embeddings, qa_from_db, doc_loader

import streamlit as st

from my_prompts import file_map, file_combine, youtube_map, youtube_combine

import os

from summary_utils import doc_to_text, token_counter, summary_prompt_creator, doc_to_final_summary


def pdf_to_text(pdf_file):
    """
    Convert a PDF file to a string of text.

    :param pdf_file: The PDF file to convert.

    :return: A string of text.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = StringIO()
    for i in range(len(pdf_reader.pages)):
        p = pdf_reader.pages[i]
        text.write(p.extract_text())
    return text.getvalue().encode('utf-8')


def check_gpt_4():
    """
    Check if the user has access to GPT-4.

    :param api_key: The user's OpenAI API key.

    :return: True if the user has access to GPT-4, False otherwise.
    """
    try:
        ChatOpenAI(model_name='gpt-4').call_as_llm('Hi')
        return True
    except Exception as e:
        return False



def token_limit(doc, maximum=200000):
    """
    Check if a document has more tokens than a specified maximum.

    :param doc: The langchain Document object to check.

    :param maximum: The maximum number of tokens allowed.

    :return: True if the document has less than the maximum number of tokens, False otherwise.
    """
    text = doc_to_text(doc)
    count = token_counter(text)
    print(count)
    if count > maximum:
        return False
    return True


def token_minimum(doc, minimum=2000):
    """
    Check if a document has more tokens than a specified minimum.

    :param doc: The langchain Document object to check.

    :param minimum: The minimum number of tokens allowed.

    :return: True if the document has more than the minimum number of tokens, False otherwise.
    """
    text = doc_to_text(doc)
    count = token_counter(text)
    if count < minimum:
        return False
    return True


def validate_api_key(model_name='gpt-3.5-turbo'):
    try:
        load_dotenv('test.env')
        print(os.getenv('OPENAI_API_KEY'))
        ChatOpenAI(model_name=model_name).call_as_llm('Hi')
        print('API Key is valid')
        return True
    except Exception as e:
        print(e)
        st.warning('API key is invalid or OpenAI is having issues.')
        print('Invalid API key.')


def create_chat_model_for_summary(api_key, use_gpt_4):
    """
    Create a chat model ensuring that the token limit of the overall summary is not exceeded - GPT-4 has a higher token limit.

    :param api_key: The OpenAI API key to use for the chat model.

    :param use_gpt_4: Whether to use GPT-4 or not.

    :return: A chat model.
    """
    if use_gpt_4:
        return ChatOpenAI(openai_api_key=api_key, temperature=0, max_tokens=500, model_name='gpt-3.5-turbo')
    else:
        return ChatOpenAI(openai_api_key=api_key, temperature=0, max_tokens=250, model_name='gpt-3.5-turbo')


def process_summarize_button(file_or_transcript, api_key, use_gpt_4, find_clusters, file=True):
    """
    Processes the summarize button, and displays the summary if input and doc size are valid

    :param file_or_transcript: The file uploaded by the user or the transcript from the YouTube URL

    :param api_key: The API key entered by the user

    :param use_gpt_4: Whether to use GPT-4 or not

    :param find_clusters: Whether to find optimal clusters or not, experimental

    :return: None
    """
    if not validate_input(file_or_transcript, api_key, use_gpt_4):
        return

    with st.spinner("Summarizing... please wait..."):

        if file:
            doc = doc_loader(file_or_transcript)
            map_prompt = file_map
            combine_prompt = file_combine
            head, tail = os.path.split(file_or_transcript)
            name = tail.split('.')[0]

        else:
            doc = file_or_transcript
            map_prompt = youtube_map
            combine_prompt = youtube_combine
            name = str(file_or_transcript)[30:44].strip()

        llm = create_chat_model_for_summary(api_key, use_gpt_4)
        initial_prompt_list = summary_prompt_creator(map_prompt, 'text', llm)
        final_prompt_list = summary_prompt_creator(combine_prompt, 'text', llm)

        if not validate_doc_size(doc):
            return

        if find_clusters:
            summary = doc_to_final_summary(doc, 10, initial_prompt_list, final_prompt_list, api_key, use_gpt_4, find_clusters)

        else:
            summary = doc_to_final_summary(doc, 10, initial_prompt_list, final_prompt_list, api_key, use_gpt_4)

        st.markdown(summary, unsafe_allow_html=True)
        with open(f'summaries/{name}_summary.txt', 'w') as f:
            f.write(summary)
        st.markdown(f' # Summary saved as {name}_summary.txt')




def validate_doc_size(doc):
    """
    Validates the size of the document

    :param doc: doc to validate

    :return: True if the doc is valid, False otherwise
    """
    if not token_limit(doc, 800000):
        st.warning('File or transcript too big!')
        return False

    if not token_minimum(doc, 2000):
        st.warning('File or transcript too small!')
        return False
    return True


def validate_input(file_or_transcript, use_gpt_4):
    """
    Validates the user input, and displays warnings if the input is invalid

    :param file_or_transcript: The file uploaded by the user or the YouTube URL entered by the user

    :param api_key: The API key entered by the user

    :param use_gpt_4: Whether the user wants to use GPT-4

    :return: True if the input is valid, False otherwise
    """
    if file_or_transcript == None:
        st.warning("Please upload a file or enter a YouTube URL.")
        return False

    if not validate_api_key():
        st.warning('Key not valid or API is down.')
        return False

    if use_gpt_4 and not check_gpt_4():
        st.warning('Key not valid for GPT-4.')
        return False

    return True


def generate_answer(db=None, llm_model=None):
    user_message = st.session_state.text_input
    if db and user_message.strip() != "":
        with st.spinner('Generating answer...'):
            print('About to call API')
            sys_message = qa_from_db(user_message, db, llm_model)
            print('Done calling API')
            st.session_state.history.append({'message': user_message, 'is_user': True})
            st.session_state.history.append({'message': sys_message, 'is_user': False})
    else:
        print(user_message)
        print('failed')
        print(db)

def load_db_from_file_and_create_if_not_exists(file_path):
    with st.spinner('Loading chat embeddings...'):
        try:
            db = load_chat_embeddings(file_path)
            print('success')
        except RuntimeError:
            print('not found')
            create_and_save_chat_embeddings(file_path)
            db = load_chat_embeddings(file_path)
    if db:
        st.success('Loaded successfully! Start a chat below.')
    else:
        st.warning('Something went wrong... failed to load chat embeddings.')
    return db