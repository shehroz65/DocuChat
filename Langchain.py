#!/usr/bin/env python
# coding: utf-8


# In[ ]:

import os
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import OnlinePDFLoader, UnstructuredPDFLoader, PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain


# In[ ]:




# In[ ]:




model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda",
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)


# In[ ]:


llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})


# In[ ]:


loader = PyPDFLoader("Q.pdf")
pdfData = loader.load()

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
splitData = text_splitter.split_documents(pdfData)
print ("Split Data is", splitData)

# In[ ]:





# In[ ]:


collection_name = "clarett_collection"
local_directory = "clarett_vect_embedding"
persist_directory = os.path.join(os.getcwd(), local_directory)

embeddings = HuggingFaceEmbeddings()
vectDB = Chroma.from_documents(splitData,
                      embeddings,
                      collection_name=collection_name,
                      persist_directory=persist_directory
                      )
vectDB.persist()


# In[ ]:


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chatQA = ConversationalRetrievalChain.from_llm(
            llm,
            vectDB.as_retriever(),
            memory=memory)

print ("Memory is", memory)
print ("Chatqa is ", chatQA)
# In[ ]:


chat_history = []
qry = ""
while qry != 'done':
    qry = input('Question: ')
    if qry != exit:
        print ("Gettting response")
        response = chatQA({"question": qry, "chat_history": chat_history})
        
        print(response)
        print ("GOT RESPONSE")


# In[ ]:


#

