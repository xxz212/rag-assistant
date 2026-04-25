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

# ── 根据目标等级定制 prompt ────────────────────────────────────────────────────
LEVEL_PROMPTS = {
    "passed": """你是一个知识助手，只根据以下文档内容回答问题。
如果文档中没有相关信息，请说"文档中未找到相关信息"。

回答要求：简洁通俗，直接回答，避免专业术语，100字以内。

文档内容：
{context}

问题：{question}

请按以下格式回答：
## 📌 答案
用1-2句话直接回答。
""",

    "credit": """你是一个知识助手，只根据以下文档内容回答问题。
如果文档中没有相关信息，请说"文档中未找到相关信息"。

回答要求：清晰准确，有条理，适当使用要点列举。

文档内容：
{context}

问题：{question}

请按以下格式回答：
## 📌 答案摘要
一句话概括。

## 📋 详细说明
- 关键要点1
- 关键要点2
- 关键要点3
""",

    "distinction": """你是一个知识助手，只根据以下文档内容回答问题。
如果文档中没有相关信息，请说"文档中未找到相关信息"。

回答要求：详细全面，重点突出，关键术语加粗。

文档内容：
{context}

问题：{question}

请按以下格式回答：
## 📌 答案摘要
一句话概括核心。

## 📋 详细说明
- 用要点列出关键信息
- **重要术语或数据**请加粗

## ⚠️ 注意事项（如有）
补充说明或限制条件。
""",

    "high_distinction": """你是一个专业知识助手，只根据以下文档内容回答问题。
如果文档中没有相关信息，请说"文档中未找到相关信息"。

回答要求：深入全面，发散思维，结合实际应用场景扩展说明。

文档内容：
{context}

问题：{question}

请按以下格式回答：
## 📌 核心答案
一句话概括。

## 📋 深度解析
- **关键点1**：详细说明
- **关键点2**：详细说明
- **关键点3**：详细说明

## 🔗 延伸思考
结合实际场景，说明这个知识点的应用或关联概念。

## ⚠️ 注意事项（如有）
补充说明。
"""
}

# 出题 prompt
QUIZ_PROMPT = ChatPromptTemplate.from_template("""根据以下文档内容，出3道考题帮助理解，难度对应"{level}"等级。

文档内容：
{context}

请按以下格式输出：
## 🎯 例题练习

**题目1：**（问题）
> 💡 参考答案：（答案）

**题目2：**（问题）
> 💡 参考答案：（答案）

**题目3：**（问题）
> 💡 参考答案：（答案）
""")

def load_file(path, suffix):
    # 根据文件类型选择对应加载器
    if suffix == "pdf":
        return PyMuPDFLoader(path).load()
    elif suffix == "txt":
        return TextLoader(path).load()
    elif suffix == "docx":
        return Docx2txtLoader(path).load()

def build_qa(files):
    # 加载文件 → 分块 → 向量化 → 存入Chroma
    docs = []
    for path, suffix in files:
        docs += load_file(path, suffix)
    chunks = SPLITTER.split_documents(docs)
    db = Chroma.from_documents(chunks, DashScopeEmbeddings(dashscope_api_key=KEY))
    llm = ChatTongyi(dashscope_api_key=KEY, model_name="qwen-plus", temperature=0.1)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return retriever, llm

def ask(retriever, llm, question, level="distinction"):
    # 根据目标等级选择对应prompt，检索文档，生成回答
    prompt_template = LEVEL_PROMPTS.get(level, LEVEL_PROMPTS["distinction"])
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    sources = list({doc.metadata.get("source", "") for doc in docs})
    return answer, sources

def generate_quiz(retriever, llm, level="distinction"):
    # 从文档中随机检索内容，生成3道例题
    docs = retriever.invoke("请介绍文档的主要内容和核心概念")
    context = "\n\n".join([d.page_content for d in docs])
    chain = QUIZ_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "level": level})