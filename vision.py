import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
KEY = os.getenv("GOOGLE_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _call_gemini_vision(image_path, prompt):
    """Call Gemini 1.5 Flash with an image + text prompt, return text response."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=KEY, temperature=0.1)
    ext = Path(image_path).suffix.lower().replace(".", "")
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encode_image(image_path)}"}},
        {"type": "text", "text": prompt}
    ])
    return llm.invoke([msg]).content

def analyze_image(image_path, question="请详细描述这张图片的内容、背景故事和关键信息。"):
    return _call_gemini_vision(image_path, question)

MODE_PROMPTS = {
    "ecommerce": """你是跨境电商运营专家，请分析这张产品图，按以下格式输出：

**商品标题**（30字以内，突出核心卖点）

**核心卖点**
- 卖点1
- 卖点2
- 卖点3
- 卖点4
- 卖点5

**商品描述**（200字左右，适合详情页使用）""",

    "contract": """你是专业文件信息提取专家，请从图片中提取以下关键信息：

**文件类型**

**当事方**

**关键金额/数量**

**重要日期**

**核心条款/条件**

**其他重要信息**""",

    "competitor": """你是资深产品经理，请分析这张竞品截图：

**产品定位**

**核心功能**

**UI设计风格**

**文案策略**

**优势分析**

**劣势/可优化点**""",

    "interior": """你是专业室内设计师，请分析这张室内照片：

**现有风格判断**

**空间优化建议**

**重点改造区域**

**预算参考**（分低/中/高三档）

**软装搭配建议**""",

    "medical": """你是医学健康顾问（本分析仅供参考，不构成医疗诊断，请以医生意见为准），请分析这份报告：

**异常指标**（注明参考范围）

**正常指标确认**

**需要关注的事项**

**生活方式建议**""",
}

def analyze_image_by_mode(image_path, mode, followup=None):
    prompt = followup if followup else MODE_PROMPTS[mode]
    return _call_gemini_vision(image_path, prompt)

def analyze_image_with_context(image_path, question, doc_context=""):
    prompt = question
    if doc_context:
        prompt = f"""请结合以下文档内容和图片，回答问题。

文档内容：
{doc_context}

问题：{question}

请综合图片和文档信息给出完整回答。"""
    return _call_gemini_vision(image_path, prompt)
