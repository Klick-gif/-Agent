from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
import pickle

embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key="sk-eb8efc4f031c464c8db80722d7c7dbd1",
)

""" text_splitter = RecursiveCharacterTextSplitter.from_language(language="markdown", chunk_size=200, chunk_overlap=20)
all_texts = []
for i in range(7):
    loader1 = PyPDFLoader(f"论文{i}.pdf")
    pdf_docs = loader1.load_and_split()
    pdf_texts = text_splitter.split_documents(pdf_docs)
    all_texts += pdf_texts

loader2 = TextLoader("battery.txt", encoding="utf-8")
txt_docs = loader2.load()
txt_texts = text_splitter.split_documents(txt_docs)
all_texts += txt_texts

loader3 = Docx2txtLoader('文章1.docx')
docx_docs = loader3.load()
docx_texts = text_splitter.split_documents(docx_docs)
all_texts += docx_texts


db = FAISS.from_documents(all_texts, embeddings)
with open("battery_db", "wb") as f:
    pickle.dump(db, f) """

with open('battery_db', "rb") as f:
    db = pickle.load(f)
retriever = db.as_retriever(search_kwargs={'k': 3})
res = retriever.invoke("储能电站是什么")
print(res)