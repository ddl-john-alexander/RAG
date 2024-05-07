# RAG
This project demonstrates how to create a Retrieval Augmented Generation (RAG) solution in Domino.

## Files

*  Solution_Overview.pdf : A diagram outlining the RAG process flow. 

*  RAG.ipynb : This notebook initializes the vector store,  constructs a prompt, builds the QA chain for RAG and allows users to ask queries embeds the query,and calls OpenAI's models and the vector data store to get a response. You will need your OpenAPI and Pinecone api keys to be set in the environment for this example. To work with OpenAI Embeddings, set up your Pinecone index to have 1536 dimensions.

*  RAG_new.ipynb : This notebook cinitializes the vector store,  constructs a prompt, builds the QA chain for RAG and allows users to ask queries embeds the query,and calls model endpoints and the vector data store to get a response using these new features:  
 - Domino AI Gateway Endpoints for both embedding and chat model access
 - Domino Pinecone vector database connector for all data access
 
 You will need to create an OpenAI Domino AI Gateway Endpoint and Domino Pinecone vector data source for this example

*  Process_data.ipynb : This notebook loads and splits a PDF, converts it to embeddings, stores the embeddings in Pinecone, and runs the semantic search against the embeddings,
  
*  example_prompts.txt : Has a few examples of questions that can be presented as prompts to the QA chain
  
*  app.sh : Script required to setup and use Streamlit in Domino
  
*  streamlit_app.py : This file contains code that sets up the UI and workflow for a Streamlit chatbot. The app needs OpenAI and Pinecone API keys to set in the sidebar to run

*  apple-10K-20230930.pdf :  The Apple 10-K document to answer queries about.

On `se-demo` this was run on a `Medium` hardware tier

## Environment Setup

### Custom base image 
```Domino Standard Environment Py3.9 R4.2```


### Dockerfile instructions

```
USER root:root

RUN pip uninstall --yes mlflow

RUN pip install -q -U trl>=0.7.1 transformers>=4.35.0 accelerate>=0.23.0 peft>=0.6.0 autoawq>=0.1.6 \
datasets>=2.14.5 bitsandbytes>=0.41.1 einops>=0.6.1 evaluate>=0.4.0 langchain-anthropic \
Flask Flask-Compress Flask-Cors jsonify uWSGI \
langchain langchain-openai langchain-experimental sentence-transformers==2.3.1 ragatouille \
ipywidgets langchainhub apify-client chromadb tiktoken SQLAlchemy==2.0.1 qdrant-client mlflow[genai] \
presidio-analyzer presidio-anonymizer spacy Faker streamlit spacy pinecone-client dominodatalab-data==5.11.0.dev2 \ 
cohere langchain-cohere langchain-pinecone

RUN pip install -i https://test.pypi.org/simple/ streamlit-chat-domino
```
On `se2-demo` this environment is available as `MedRAG`

### 

