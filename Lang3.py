from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.schema import retriever
from langchain.memory import ConversationBufferMemory



loader = PyPDFLoader("Q.pdf")
data = loader.load()

#Step 05: Split the Extracted Data into Text Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

text_chunks = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

template = """Answer the question based only on the following context: {context}. Question: {question} """
prompt = ChatPromptTemplate.from_template(template)

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

#Import Model
llm = LlamaCpp(
    streaming = True,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    model_path="yarn-mistral-7b-128k.Q5_K_S.gguf",
    temperature=0.1,
    top_p=1,
    verbose=True,
    n_ctx=2048
)


#chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = ""
while (query != "exit"):
    print ("WRITE YOUR QUERY")
    query = input(str)
    chain = (
         {"context": retriever, "question": query}
         | prompt
         | llm
         | StrOutputParser())
    output = chain.invoke(query)
    print (output['result'])