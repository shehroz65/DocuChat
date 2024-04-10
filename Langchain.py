from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

import time

# from langchain.chat_models import ChatOpenAI
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

DOCUMETNS_DB_DIR = ''


class DocSearchWrapper:
    def __init__(self):
        db = Chroma(
            persist_directory=DOCUMETNS_DB_DIR,
            # embedding_function=OpenAIEmbeddings(),
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        )

        retriever = db.as_retriever(search_kwargs={"k": 3})

        model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
        #model_basename = "gptq_model-4bit-128g"

        use_triton = False

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=use_triton,
            quantize_config=None,
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=4096,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.15,
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Interactive questions and answers
        self.CRChain = ConversationalRetrievalChain.from_llm(
            # llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            # condense_question_llm=self.llm
            # condense_question_llm=ChatOpenAI(),
        )

        self.chat_history = []

    def search_docbase(self, query):
        result = self.CRChain({"question": query, "chat_history": self.chat_history})

        self.chat_history.append((query, result["answer"]))

        return result

    def clear_history(self):
        self.chat_history = []


if __name__ == "__main__":
    doc_search = DocSearchWrapper()

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query == "clear":
            doc_search.clear_history()
            continue
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = doc_search.search_docbase(query)
        print(res)

        answer, docs = res["answer"], res["source_documents"]
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        print("Sources:\n")
        for document in docs:
            # print("> " + document.metadata["source"] + f": page({document.metadata['page']})")
            print("> " + document.metadata["source"])
