from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
#from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
import os

model = ChatMistralAI(model="mistral:mistral-medium-latest")
model = ChatGroq(model="groq:gemma2-9b-it")

loader = TextLoader("healthy_aging.txt", encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = MistralAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)

def retrieve_docs(query: str):
    retriever = db.as_retriever()
    docs = retriever.invoke(query)
    return docs[0].page_content


def get_var(question, *args, **kwargs):
    context = retrieve_docs(question)
    return {"output": context}