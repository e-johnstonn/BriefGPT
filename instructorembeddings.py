# import InstructorEmbedding as ie
#
#
# model = ie.INSTRUCTOR('hkunlp/instructor-xl')

"""
Experimental alternative to using OpenAI embeddings. Large computational cost.

Uncomment, install requirements, and replace call to embed_docs in extract_summary_docs in asdasd to use.

Can use instructor models other than instructor-xl.

"""
# def embed_docs_instructor(docs, model=model):
#     """
#     Embed a list of loaded langchain Document objects into a list of vectors.
#
#     :param docs: A list of loaded langchain Document objects to embed.
#
#     :param model: The Instructor model to use.
#
#     :return: A list of vectors.
#     """
#     vectors = []
#     for x in docs:
#         vector = model.encode(x.page_content)
#         vectors.append(vector)
#     return vectors

