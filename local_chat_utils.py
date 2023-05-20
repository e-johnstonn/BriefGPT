import os

from langchain import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from summary_utils import doc_loader, remove_special_tokens

import streamlit as st
def get_embeddings(docs_lang):
    if docs_lang.upper() in ['CHINESE', 'ZH']:
        return HuggingFaceEmbeddings(model_name=os.getenv('EMBED_PATH'))
    else:
        return HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-base')

def create_and_save_local(file_path, docs_lang):
    name = os.path.split(file_path)[1].split('.')[0]
    embeddings = get_embeddings(docs_lang)
    doc = doc_loader(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(doc)
    processed_split_docs = remove_special_tokens(split_docs)
    db = FAISS.from_documents(processed_split_docs, embeddings)
    db.save_local(folder_path='local_embeddings', index_name=name)


def load_local_embeddings(file_path, docs_lang):
    name = os.path.split(file_path)[1].split('.')[0]
    embeddings = get_embeddings(docs_lang)
    db = FAISS.load_local(folder_path='local_embeddings', index_name=name, embeddings=embeddings)
    return db



def load_db_from_file_and_create_if_not_exists_local(file_path, docs_lang):
    with st.spinner('Loading chat embeddings...'):
        try:
            db = load_local_embeddings(file_path, docs_lang)
            print('success')
        except RuntimeError:
            print('not found')
            create_and_save_local(file_path, docs_lang)
            db = load_local_embeddings(file_path, docs_lang)
    if db:
        st.success('Loaded successfully! Start a chat below.')
    else:
        st.warning('Something went wrong... failed to load chat embeddings.')
    return db









