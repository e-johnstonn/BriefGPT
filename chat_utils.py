import os

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.llms import GPT4All

from fuzzywuzzy import fuzz

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from my_prompts import chat_prompt

from dotenv import load_dotenv

from summary_utils import doc_loader, remove_special_tokens

load_dotenv()

nltk.download('stopwords')
nltk.download('punkt')


def create_and_save_chat_embeddings(file_path):
    name = os.path.split(file_path)[1].split('.')[0]
    embeddings = OpenAIEmbeddings()
    doc = doc_loader(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(doc)
    processed_split_docs = remove_special_tokens(split_docs)
    db = FAISS.from_documents(processed_split_docs, embeddings)
    db.save_local(folder_path='embeddings', index_name=name)


def load_chat_embeddings(file_path):
    name = os.path.split(file_path)[1].split('.')[0]
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(folder_path='embeddings', index_name=name, embeddings=embeddings)
    return db


def results_from_db(db:FAISS, question, num_results=10):
    results = db.similarity_search(question, k=num_results)
    return results


def rerank_fuzzy_matching(question, results, num_results=4):
    filtered_question = filter_stopwords(question)
    if filtered_question == '':
        return results[-4:]
    scores_and_results = []
    for result in results:
        score = fuzz.partial_ratio(question, result.page_content)
        scores_and_results.append((score, result.page_content))

    scores_and_results.sort(key=lambda x: x[0], reverse=True)
    reranked = [result for score, result in scores_and_results]

    return reranked[:num_results]


def filter_stopwords(question):
    words = word_tokenize(question)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    filtered_sentence = ' '.join(filtered_words)
    return filtered_sentence


def qa_from_db(question, db, llm_name):
    print('getting results...')
    llm = create_llm(llm_name)
    results = results_from_db(db, question)
    reranked_results = rerank_fuzzy_matching(question, results)
    message = f'{chat_prompt} ---------- Context: {reranked_results} -------- User Question: {question} ---------- Response:'
    print(message)
    print(llm(message))
    output = llm(message)
    return output


def create_llm(llm_name):
    if type(llm_name) != str:
        return llm_name
    else:
        llm = OpenAI(model_name=llm_name)
    return llm









