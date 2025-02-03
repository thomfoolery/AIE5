import tiktoken
from operator import itemgetter

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
"""

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def create_rag_chain(llm):
    # docs = PyMuPDFLoader("./docs/DeepSeek_R1.pdf").load()
    docs = PyMuPDFLoader("https://arxiv.org/pdf/2404.19553").load()

    def tiktoken_len(text):
        tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(
            text,
        )
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 0,
        length_function = tiktoken_len,
    )

    split_chunks = text_splitter.split_documents(docs)
    
    qdrant_vectorstore = Qdrant.from_documents(
        split_chunks,
        embedding_model,
        location=":memory:",
        collection_name="extending_context_window_llama_3",
    )

    qdrant_retriever = qdrant_vectorstore.as_retriever()
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    rag_chain = (
        {
            "context": itemgetter("question") | qdrant_retriever, 
            "question": itemgetter("question")
        }
        | rag_prompt 
        | llm 
        | StrOutputParser()
    )

    return rag_chain

    #####

    # result = rag_chain.invoke({"question" : "What is different about Deepseek R1?"})

    # print("RAG Chain Result:", result)

    #####
