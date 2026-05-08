# 数据分析模块
# 支持 CSV / Excel 文件
# 功能：自然语言问数据问题 + 根据用户需求生成可下载图表

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互模式，适合服务器环境
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
KEY = os.getenv("DASHSCOPE_API_KEY")

def load_data(file_path, suffix):
    """加载 CSV 或 Excel 文件，返回 DataFrame"""
    if suffix == "csv":
        return pd.read_csv(file_path)
    elif suffix in ["xlsx", "xls"]:
        return pd.read_excel(file_path)

def df_summary(df):
    """生成数据概览字符串，供LLM理解数据结构"""
    return f"""数据维度：{df.shape[0]}行 x {df.shape[1]}列
列名：{list(df.columns)}
数据类型：{df.dtypes.to_dict()}
前5行数据：
{df.head().to_string()}
基础统计：
{df.describe().to_string()}"""

def ask_data(df, question):
    """用自然语言问数据问题，返回文字回答"""
    llm = ChatTongyi(dashscope_api_key=KEY, model_name="qwen-plus", temperature=0.1)
    prompt = ChatPromptTemplate.from_template("""你是一个数据分析助手。
根据以下数据信息回答用户问题，给出准确的数据统计结论。

数据信息：
{summary}

问题：{question}

请给出清晰的数据分析结论，数字要准确。""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"summary": df_summary(df), "question": question})

def generate_chart(df, instruction, save_path="chart.png"):
    """
    根据用户指令生成图表并保存为PNG
    instruction：用户的图表需求，例如"画一个各地区销售额的柱状图"
    返回：保存的图片路径
    """
    # 先让LLM生成matplotlib代码
    llm = ChatTongyi(dashscope_api_key=KEY, model_name="qwen-plus", temperature=0.1)
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

    code = chain = prompt | llm | StrOutputParser()
    code = chain.invoke({
        "summary": df_summary(df),
        "instruction": instruction,
        "save_path": save_path,
        "data_preview": df.head(10).to_string()
    })

    # 清理代码（去掉可能的markdown标记）
    code = code.replace("```python", "").replace("```", "").strip()

    # 执行生成的代码
    exec(code, {"plt": plt, "pd": pd})
    return save_path