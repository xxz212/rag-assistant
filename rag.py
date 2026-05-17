import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
KEY = os.getenv("GOOGLE_API_KEY")

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

# ── 根据目标等级定制 prompt ────────────────────────────────────────────────────
LEVEL_PROMPTS = {
    "passed": """You are a knowledge assistant. Answer questions based solely on the document content below.
If the information is not found in the document, say "This information is not available in the document."

Response style: concise and plain, answer directly, avoid jargon, within 100 words.

Document content:
{context}

Question: {question}

Respond in the following format:
## 📌 Answer
Answer in 1–2 sentences.
""",

    "credit": """You are a knowledge assistant. Answer questions based solely on the document content below.
If the information is not found in the document, say "This information is not available in the document."

Response style: clear and structured, use bullet points where appropriate.

Document content:
{context}

Question: {question}

Respond in the following format:
## 📌 Summary
One sentence overview.

## 📋 Details
- Key point 1
- Key point 2
- Key point 3
""",

    "distinction": """You are a knowledge assistant. Answer questions based solely on the document content below.
If the information is not found in the document, say "This information is not available in the document."

Response style: thorough and detailed, highlight key terms in bold.

Document content:
{context}

Question: {question}

Respond in the following format:
## 📌 Summary
One sentence core answer.

## 📋 Details
- List key information as bullet points
- **Important terms or data** should be bolded

## ⚠️ Notes (if applicable)
Additional context or limitations.
""",

    "high_distinction": """You are a professional knowledge assistant. Answer questions based solely on the document content below.
If the information is not found in the document, say "This information is not available in the document."

Response style: in-depth, analytical, connect to real-world applications.

Document content:
{context}

Question: {question}

Respond in the following format:
## 📌 Core Answer
One sentence summary.

## 📋 Deep Analysis
- **Key point 1**: detailed explanation
- **Key point 2**: detailed explanation
- **Key point 3**: detailed explanation

## 🔗 Extended Thinking
Connect this concept to real-world applications or related ideas.

## ⚠️ Notes (if applicable)
Additional context.
"""
}

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
    if suffix == "pdf":
        return PyMuPDFLoader(path).load()
    elif suffix == "txt":
        return TextLoader(path).load()
    elif suffix == "docx":
        return Docx2txtLoader(path).load()

def build_qa(files):
    docs = []
    for path, suffix in files:
        docs += load_file(path, suffix)
    chunks = SPLITTER.split_documents(docs)
    db = Chroma.from_documents(
        chunks,
        GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=KEY)
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=KEY, temperature=0.1)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return retriever, llm

def ask(retriever, llm, question, level="distinction"):
    prompt_template = LEVEL_PROMPTS.get(level, LEVEL_PROMPTS["distinction"])
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    sources = list({doc.metadata.get("source", "") for doc in docs})
    return answer, sources

def generate_quiz(retriever, llm, level="distinction"):
    docs = retriever.invoke("请介绍文档的主要内容和核心概念")
    context = "\n\n".join([d.page_content for d in docs])
    chain = QUIZ_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "level": level})
