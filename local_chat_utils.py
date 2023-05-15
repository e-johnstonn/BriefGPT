import os

from langchain import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from summary_utils import doc_loader, remove_special_tokens

import streamlit as st


def create_and_save_local(file_path, model_name):
    name = os.path.split(file_path)[1].split('.')[0]
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    doc = doc_loader(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(doc)
    processed_split_docs = remove_special_tokens(split_docs)
    db = FAISS.from_documents(processed_split_docs, embeddings)
    db.save_local(folder_path='local_embeddings', index_name=name)


def load_local_embeddings(file_path, model_name):
    name = os.path.split(file_path)[1].split('.')[0]
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    db = FAISS.load_local(folder_path='local_embeddings', index_name=name, embeddings=embeddings)
    return db



def load_db_from_file_and_create_if_not_exists_local(file_path, model_name):
    with st.spinner('Loading chat embeddings...'):
        try:
            db = load_local_embeddings(file_path, model_name)
            print('success')
        except RuntimeError:
            print('not found')
            create_and_save_local(file_path, model_name)
            db = load_local_embeddings(file_path, model_name)
    if db:
        st.success('Loaded successfully! Start a chat below.')
    else:
        st.warning('Something went wrong... failed to load chat embeddings.')
    return db









