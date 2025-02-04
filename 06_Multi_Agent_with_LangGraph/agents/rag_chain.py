import tiktoken
from operator import itemgetter

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import ArxivQueryRun
import arxiv

RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
"""

class GetArchivePaperUrl(ArxivQueryRun):
    def _run(self, query: str) -> str:
        """Return URL of Arxiv paper"""
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=1)
        for result in client.results(search):
            return f"http://arxiv.org/pdf/{result.get_short_id()}.pdf"
        return "No results found"
    
archive_tool = GetArchivePaperUrl()
url_chain = (lambda x: archive_tool.run({"query": x["query"]}))

def get_dynamic_rag_chain():
    paper_url = url_chain({"query": "get the url of the paper about deepseek r1"})
    
    return paper_url

def create_rag_chain(llm, paper_url = "https://arxiv.org/pdf/2404.19553"):
    docs = PyMuPDFLoader(paper_url).load()
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

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
