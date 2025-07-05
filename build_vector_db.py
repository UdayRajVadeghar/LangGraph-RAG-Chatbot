import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

text = open("uday_info.txt", "r", encoding="utf-8").read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([text])

vectorstore = Chroma.from_documents(
    docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    persist_directory="./chroma_db"
)

vectorstore.persist()
print("✅ Vector DB created at ./chroma_db")
