# BriefGPT

BriefGPT is a powerful, locally-run tool for document summarization and querying using OpenAI's models. You retain **full control** over your documents and API keys, ensuring privacy and security.

# Example (using the "Sparks of AGI" paper, sped up)
![chat](https://i.imgur.com/ipgvsgb.gif)




# Setup
1. Clone the repository
2. Download all requirements
``pip install -r requirements.txt``
3. Set your API key in test.env
4. Navigate to the project directory and run
```streamlit run main.py```
5. Add your PDF's or .txt's to the documents folder in the project directory




# How it works
## Chat
1. Creating and saving embeddings - once you load a file, it is broken into chunks and stored as a FAISS index in the 'embeddings' folder. These embeddings will be used if you load the document into the chat again.
2. Retrieving, ranking, and processing results - a similarity search is performed on the index to get the top n results. These results are then re-ranked by a function that strips the original query of stopwords and uses fuzzy matching to find the similarity in exact words between the query and the retrieved results. This gets better results than solely doing a similarity search.
3. Output - the re-ranked results and the user query are passed to the llm, and the response is displayed.




## Summarization
1. Input - can handle both documents and YouTube URL's - will find the transcript and generate a summary based off of that.
2.  Processing and embedding - before embedding, documents are stripped of any special tokens that might cause errors. Documents are embedded in chunks of varying size, depending on the overall document's size. 
3. Clustering - once the documents are embedded, they are grouped into clusters using the K-means algorithm. The number of clusters can be predetermined (10) or variable (finds optimal number based on the elbow method). The embedding closest to each cluster centroid is retrieved - each cluster might represent a different theme or idea, and the retrieved embeddings are those that best encapsulate that theme or idea - that's the goal, at least.
4. Summarization - summarization is performed in two steps. First, each retrieved embedding is matched with its corresponding text chunk. Each chunk is passed to GPT-3.5 in an individual call to the API - these calls are made in parallel. Once we have accumulated a summary for each chunk, the summaries are passed to GPT-3.5 or GPT-4 for the final summary.
5. Output - the summary is displayed on the page and saved as a text file. 
![summary](https://i.imgur.com/sUcay6a.gif)

Support for locally run LLM's is coming. 

Built using Langchain! This is project was made for fun, and is likely full of bugs. It is not fully optimized. Contributions or bug reports are welcomed!

todo: keep summary in session state, save transcripts when loaded to summarize, local llm support (fully offline)
