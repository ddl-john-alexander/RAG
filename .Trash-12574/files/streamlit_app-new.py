import os
import pickle
import random
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.pinecone import Pinecone 
import pinecone

from streamlit.web.server import websocket_headers
from streamlit_chat import message

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_API_ENV')

#Create prompt template
prompt_template = """Use the following pieces of context to answer the question enclosed within  3 backticks at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Please provide an answer which is factually correct and based on the information retrieved from the vector store.
Please also mention any quotes supporting the answer if any present in the context supplied within two double quotes "" .

{context}

QUESTION:```{question}```
ANSWER:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
#
chain_type_kwargs = {"prompt": PROMPT}

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

st.set_page_config(initial_sidebar_state='collapsed')

clear_button = st.sidebar.button("Clear Conversation", key="clear")

qa_chain = None
doc_store = None

if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
 

model = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model,
    openai_api_key=OPENAI_API_KEY
)

text_field = "symptoms"
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = "medical-qa-search"
index = pinecone.Index(index_name)

# switch back to normal index for langchain
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

rag_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4',
    temperature=0.0
)
    
qa_chain = RetrievalQA.from_chain_type(llm=rag_llm,
                                       chain_type="stuff",
                                       chain_type_kwargs={"prompt": PROMPT},
                                       retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                                       return_source_documents=True
                                      )

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
    if submit_button and user_input and qa_chain:
        answer = None
        with st.spinner("Searching for the answer..."):
            result = qa_chain(user_input)
        if result:
            answer = result["result"]
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(answer)
        
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, logo='https://freesvg.org/img/1367934593.png', key=str(i) + '_user')
                message(st.session_state["generated"][i], logo='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQk6e8aarUy37BOHMTSk-TUcs4AyAy3pfAHL-F2K49KHNEbI0QUlqWJFEqXYQvlBdYMMJA&usqp=CAU', key=str(i))
