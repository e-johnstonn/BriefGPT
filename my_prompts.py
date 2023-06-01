file_map = """
You will be given a single section from a text. This will be enclosed in triple backticks.
Please provide a cohesive summary of the following section excerpt, focusing on the key points and main ideas, while maintaining clarity and conciseness.

'''{text}'''

FULL SUMMARY:
"""


file_combine = """
Read all the provided summaries from a larger document. They will be enclosed in triple backticks. 
Determine what the overall document is about and summarize it with this information in mind.
Synthesize the info into a well-formatted easy-to-read synopsis, structured like an essay that summarizes them cohesively. 
Do not simply reword the provided text. Do not copy the structure from the provided text.
Avoid repetition. Connect all the ideas together.
Preceding the synopsis, write a short, bullet form list of key takeaways.
Format in HTML. Text should be divided into paragraphs. Paragraphs should be indented. 

'''{text}'''


"""

youtube_map = """
You will be given a single section from a transcript of a youtube video. This will be enclosed in triple backticks.
Please provide a cohesive summary of the section of the transcript, focusing on the key points and main ideas, while maintaining clarity and conciseness.

'''{text}'''

FULL SUMMARY:
"""


youtube_combine = """
Read all the provided summaries from a youtube transcript. They will be enclosed in triple backticks.
Determine what the overall video is about and summarize it with this information in mind. 
Synthesize the info into a well-formatted easy-to-read synopsis, structured like an essay that summarizes them cohesively. 
Do not simply reword the provided text. Do not copy the structure from the provided text.
Avoid repetition. Connect all the ideas together.
Preceding the synopsis, write a short, bullet form list of key takeaways.
Format in HTML. Text should be divided into paragraphs. Paragraphs should be indented. 

'''{text}'''


"""

chat_prompt = """
You will be provided some context from a document.
Based on this context, answer the user question.
Only answer based on the given context.
If you cannot answer, say 'I don't know' and recommend a different question.

"""

hypothetical_prompt = """
Given the user's question, please generate a response that mimics the exact format in which the relevant information would appear within a document, even if the information does not exist.
The response should not offer explanations, context, or commentary, but should emulate the precise structure in which the answer would be found in a hypothetical document. 
Factuality is not important, the priority is the hypothetical structure of the excerpt. Use made-up facts to emulate the structure. 
For example, if the user question is "who are the authors?", the response should be something like
'Authors: John Smith, Jane Doe, and Bob Jones'
The user's question is:

"""

