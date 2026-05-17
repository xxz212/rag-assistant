import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
KEY = os.getenv("GOOGLE_API_KEY")

def load_data(file_path, suffix):
    if suffix == "csv":
        return pd.read_csv(file_path)
    elif suffix in ["xlsx", "xls"]:
        return pd.read_excel(file_path)

def df_summary(df):
    return f"""数据维度：{df.shape[0]}行 x {df.shape[1]}列
列名：{list(df.columns)}
数据类型：{df.dtypes.to_dict()}
前5行数据：
{df.head().to_string()}
基础统计：
{df.describe().to_string()}"""

def ask_data(df, question):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=KEY, temperature=0.1)
    prompt = ChatPromptTemplate.from_template("""你是一个数据分析助手。
根据以下数据信息回答用户问题，给出准确的数据统计结论。

数据信息：
{summary}

问题：{question}

请给出清晰的数据分析结论，数字要准确。""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"summary": df_summary(df), "question": question})

def generate_chart(df, instruction, save_path="chart.png"):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=KEY, temperature=0.1)
    prompt = ChatPromptTemplate.from_template("""你是一个数据可视化专家。
根据以下数据信息和用户需求，生成 matplotlib 绘图代码。

数据信息：
{summary}

用户需求：{instruction}

要求：
1. 代码中数据直接用 Python 字典或列表硬编码（不要读文件）
2. 图表要有标题、坐标轴标签
3. 使用 plt.savefig("{save_path}") 保存
4. 不要调用 plt.show()
5. 只输出纯Python代码，不要任何解释或markdown格式

数据前10行供参考：
{data_preview}""")

    chain = prompt | llm | StrOutputParser()
    code = chain.invoke({
        "summary": df_summary(df),
        "instruction": instruction,
        "save_path": save_path,
        "data_preview": df.head(10).to_string()
    })
    code = code.replace("```python", "").replace("```", "").strip()
    exec(code, {"plt": plt, "pd": pd})
    return save_path
