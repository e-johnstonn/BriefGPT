# BriefGPT

BriefGPT is a powerful, locally-run tool for document summarization and querying using OpenAI's models. You retain control over your documents and API keys, ensuring privacy and security.

## Update
Added support for fully local use! Instructor is used to embed documents, and the LLM can be either LlamaCpp or GPT4ALL, ggml formatted. Put your model in the 'models' folder, set up your environmental variables (model type and path), and run ```streamlit run local_app.py``` to get started. Tested with the following models: [Llama](https://huggingface.co/eachadea/ggml-vicuna-13b-1.1/blob/main/ggml-old-vic13b-q5_0.bin), [GPT4ALL](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin), [Chinese-LLaMA-Alpaca](https://huggingface.co/HardSoft2023/chinese-alpaca-plus-lora-7b-llama.cpp-q4_0.bin).

Please note this is experimental - it will be significantly slower and the quality may vary. PR's welcome!

# Example 1 (using the "Sparks of AGI" paper, sped up)
![chat](https://i.imgur.com/ipgvsgb.gif)
# Example 2 (Chinese Demo，关于李白)
<img src="https://wechatlongterm.oss-cn-beijing.aliyuncs.com/%E5%9B%9E%E7%AD%94%E7%9A%84%E9%9D%9E%E5%B8%B8%E5%A5%BD.png" alt="回答的非常好" style="zoom:33%;" />



# Setup
1. Clone the repository
2. Download all requirements
``pip install -r requirements.txt``
3. Set your API key in test.env
4. Navigate to the project directory and run
```streamlit run main.py```
5. Add your PDF's or .txt's to the documents folder in the project directory
6. If using epubs, ensure you have pandoc installed and added to PATH




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

Improved support for locally run LLM's is coming. 

Built using Langchain! This is project was made for fun, and is likely full of bugs. It is not fully optimized. Contributions or bug reports are welcomed!

todo: keep summary in session state, save transcripts when loaded to summarize
