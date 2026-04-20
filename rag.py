import os
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
KEY = os.getenv("DASHSCOPE_API_KEY")

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# 把pdf，txt，docx文件翻译成python和langchain可以理解的统一语言。
def load_file(path, suffix):
    if suffix == "pdf":
        return PyMuPDFLoader(path).load()
    elif suffix == "txt":
        return TextLoader(path).load()
    elif suffix == "docx":
        return Docx2txtLoader(path).load()

PROMPT = ChatPromptTemplate.from_template("""你是一个知识助手，只根据以下文档内容回答问题。
如果文档中没有相关信息，请说"文档中未找到相关信息"。

文档内容：
{context}

问题：{question}
""")

def build_qa(files):
    docs = []
    for path, suffix in files:
        docs += load_file(path, suffix)
    # 将大文件切块成800字的小块，为后面检索提供更精准的服务，给它们做索引，不只是记录书名
    chunks = SPLITTER.split_documents(docs)
    # 向量化embedding：调用Dashscope接口，将文件块变成高维向量
    # 入库vector store:把向量存进Chroma数据库。此时，文字已不再是文字，而是数学坐标。
    db = Chroma.from_documents(chunks, DashScopeEmbeddings(dashscope_api_key=KEY))
    retriever = db.as_retriever(search_kwargs={"k": 4})
    # 调用大模型
    llm = ChatTongyi(dashscope_api_key=KEY, model_name="qwen-plus", temperature=0.1)
    # context retriever 检索背景 question RunnablePassthrough():和向量库传递问题 strOutputParser():将AI返回的复杂对象，解析器简化成人类可以读懂的纯字符串
    chain = {"context": retriever, "question": RunnablePassthrough()} | PROMPT | llm | StrOutputParser()
    return chain, retriever
# 用户与系统的接口，负责执行查询并收回答案和证据
def ask(chain, retriever, question):
    #获取ai的回答
    answer = chain.invoke(question)
    #获取检索到的原始文档块
    docs = retriever.invoke(question)
    sources = list({doc.metadata.get("source", "") for doc in docs})
    return answer, sources