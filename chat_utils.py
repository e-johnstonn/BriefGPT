import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI


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


def rerank_fuzzy_matching(question, results, num_results=5):
    filtered_question = filter_stopwords(question)
    if filtered_question == '':
        return results[-5:]
    scores_and_results = []
    for result in results:
        score = fuzz.partial_ratio(question, result.page_content)
        scores_and_results.append((score, result))

    scores_and_results.sort(key=lambda x: x[0], reverse=True)
    reranked = [result for score, result in scores_and_results]

    return reranked[:num_results]


def filter_stopwords(question):
    words = word_tokenize(question)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    filtered_sentence = ' '.join(filtered_words)
    return filtered_sentence


def qa_from_db(question, db, llm_name):
    llm = create_llm(llm_name)
    results = results_from_db(db, question)
    reranked_results = rerank_fuzzy_matching(question, results)
    reranked_content = [result.page_content for result in reranked_results]
    if type(llm_name) != str:
        reranked_results = reranked_results[:2]
        message = f'Answer the user question based on the context. Question: {question} Context: {reranked_content[:2]} Answer:'
    else:
        message = f'{chat_prompt} ---------- Context: {reranked_content} -------- User Question: {question} ---------- Response:'
    formatted_sources = source_formatter(reranked_results)
    output = llm(message)
    return output, formatted_sources



def source_formatter(sources):
    formatted_strings = []
    for doc in sources:
        source_name = doc.metadata['source'].split('\\')[-1]
        source_content = doc.page_content.replace('\n', ' ')  # Replacing newlines with spaces
        formatted_string = f"Source name: {source_name} | Source content: '{source_content}' - end of content"
        formatted_strings.append(formatted_string)
    final_string = '\n\n\n'.join(formatted_strings)
    return final_string

def create_llm(llm_name):
    if type(llm_name) != str:
        return llm_name
    else:
        llm = OpenAI(model_name=llm_name)
    return llm









