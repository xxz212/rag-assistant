# Session Archive — RAG Knowledge Assistant

> **用途 / Purpose**: 供下次对话开始时快速恢复上下文。阅读此文档可完整还原项目现状、已做的事、未完成的事。
>
> **Last updated**: 2026-05-17（本次会话全部完成，剩余：单元测试 + CI，下次再做）

---

## 1. 项目基本信息

| 项 | 内容 |
|---|---|
| 项目名 | RAG Knowledge Assistant |
| 作者 | Nate（xxz212）|
| 身份 | UOW Master of IT 准毕业生，求职目标：墨尔本/悉尼 AI Application Engineer |
| GitHub | https://github.com/xxz212/rag-assistant |
| 当前分支 | `dev`（主分支为 `main`，生产部署用 `dev`）|
| 部署平台 | Streamlit Cloud（免费）|
| 本地路径 | `E:\Natespersonalbusinessappproject\RAG-project` |
| Python版本 | 3.11（`.python-version` 文件）|

---

## 2. 项目功能概览

三个核心功能，通过首页轮播卡片进入：

| 功能 | 入口 | 核心文件 |
|---|---|---|
| Document Q&A | 📄 Doc 卡片 | `rag.py` |
| Image Analysis | 🖼️ Image 卡片 | `vision.py` |
| Data Analysis | 📊 Data 卡片 | `data_analysis.py` |

**RAG 流程**：文件上传 → LangChain 分块 → Google Embedding (embedding-001) → Chroma 向量库 → 检索 → Gemini 1.5 Flash 生成回答

---

## 3. 技术栈（当前版本）

| 层 | 技术 |
|---|---|
| 前端 | Streamlit 1.56.0 |
| LLM | Google Gemini 1.5 Flash (`gemini-1.5-flash`) |
| 视觉 | Google Gemini 1.5 Flash（多模态）|
| Embedding | `models/embedding-001`（Google）|
| 向量库 | ChromaDB 1.5.8（本地）|
| RAG 框架 | LangChain 1.2.15 + langchain-chroma 1.1.0 |
| LangChain Gemini 桥接 | `langchain-google-genai==4.2.2` |

**之前用的是 DashScope/Qwen，本次全部换成了 Google Gemini。**

---

## 4. API Key 说明

**只有一个 key 需要填写：`GOOGLE_API_KEY`**

- 本地：在项目根目录的 `.env` 文件第2行添加：
  ```
  GOOGLE_API_KEY="你的key"
  ```
- Streamlit Cloud：Settings → Secrets → 添加：
  ```toml
  GOOGLE_API_KEY = "你的key"
  ```
- 申请地址：https://aistudio.google.com（免费，无需信用卡）
- `.env` 已在 `.gitignore` 中，不会上传到 GitHub，安全。

---

## 5. 文件结构

```
rag-assistant/
├── app.py              # Streamlit UI + session管理（主文件）
├── rag.py              # RAG pipeline（Gemini版）
├── vision.py           # 图片分析（Gemini视觉版）
├── data_analysis.py    # CSV/Excel分析 + 图表生成
├── session.py          # session state管理
├── requirements.txt    # 已锁版本
├── .env.example        # API key模板（只需填GOOGLE_API_KEY）
├── .env                # 本地key（不提交git）
├── .gitignore          # 包含.env / .venv / __pycache__
├── .python-version     # 3.11
├── README.md           # 已更新为Gemini版本
└── SESSION_LOG.md      # 本文档
```

---

## 6. 本次会话完成的所有工作

### 6.1 Bug 修复

| Bug | 原因 | 修复方式 |
|---|---|---|
| 侧边栏折叠时"双箭头"按钮几乎不可见 | CSS未命中正确的testid；且用的是fill而非color | 找到 `stExpandSidebarButton` testid，用 `color: #ffffff !important` |
| 进入功能后侧边栏消失 | 首页用了 `display:none` CSS hack，触发Streamlit折叠状态机 | 改为Python条件渲染：`if not st.session_state.show_landing:` 包住整个 `with st.sidebar:` |
| Home按钮渲染在主区域 | 缩进错误，在 `with st.sidebar:` 块外面 | 移入sidebar块内 |

**关键技术发现**：Streamlit 1.56.0 的展开按钮 testid 是 `stExpandSidebarButton`（不是 `collapsedControl`）。通过 grep `.venv/Lib/site-packages/streamlit/static/static/js/` 的 JS bundle 找到的。Streamlit 用的是 Material icon font，颜色用 `color` 属性而不是 SVG `fill`。

### 6.2 LLM 迁移：DashScope/Qwen → Google Gemini 1.5 Flash

**修改的文件：**

**`rag.py`** — 完整替换：
```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
KEY = os.getenv("GOOGLE_API_KEY")
# build_qa() 中：
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=KEY, temperature=0.1)
```

**`vision.py`** — 完整替换：
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
KEY = os.getenv("GOOGLE_API_KEY")

def _call_gemini_vision(image_path, prompt):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=KEY, temperature=0.1)
    ext = Path(image_path).suffix.lower().replace(".", "")
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encode_image(image_path)}"}},
        {"type": "text", "text": prompt}
    ])
    return llm.invoke([msg]).content
```

**`data_analysis.py`** — LLM 部分替换：
```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=KEY, temperature=0.1)
```

### 6.3 错误处理

在 `app.py` 的所有 AI 调用点加了 try/except + `st.error()`，共 6 处：
- `build_qa()` 调用（文档处理）
- `ask()` 调用（问答）
- `generate_quiz()` 调用（出题）
- `analyze_image_by_mode()` 自动分析
- `analyze_image_by_mode()` 追问
- `generate_chart()` / `ask_data()` 数据分析

### 6.4 requirements.txt 版本锁定

当前内容（已修复依赖冲突后的最终版本）：
```
streamlit==1.56.0
langchain==1.2.15
langchain-community==0.4.1
langchain-chroma==1.1.0
langchain-text-splitters==1.1.2
langchain-google-genai==4.2.2
chromadb==1.5.8
PyMuPDF==1.27.2.2
docx2txt==0.9
python-dotenv==1.2.2
protobuf==5.29.6
pillow==12.2.0
pandas==2.3.3
matplotlib==3.10.9
openpyxl==3.1.5
```

**注意**：`google-genai` 和 `google-generativeai` 不需要单独 pin，`langchain-google-genai==4.2.2` 会自动拉取兼容版本（`google-genai>=1.65.0,<2.0.0`）。

### 6.5 README 更新

完整替换为 Gemini 版本：
- 移除所有 DashScope/Qwen 引用
- 技术栈表格改为 Gemini
- 架构图改为 Google Gemini API
- Quick Start 改为 `GOOGLE_API_KEY`
- Why 部分改为"Why Google Gemini 1.5 Flash?"
- Roadmap：Streamlit Cloud 部署标记为 ✅ 完成

### 6.6 .env.example 新建

```
GOOGLE_API_KEY="your-google-gemini-api-key-here"
```

### 6.7 Streamlit Cloud 部署排错（三轮）

| 轮次 | 错误 | 修复 |
|---|---|---|
| 第1轮 | installer returned a non-zero exit code | 原因不明，开始排查 |
| 第2轮 | `protobuf==6.33.6` 不存在/冲突；`google-generativeai` 与 `google-genai` 冲突 | 改 protobuf 为 `5.29.6`，删除 `google-generativeai` pin |
| 第3轮 | `langchain-google-genai==4.2.2` 要求 `google-genai<2.0.0`，但我们 pin 了 `==2.3.0` | 删除 `google-genai==2.3.0` pin，让 langchain 自动解析 |

---

## 7. Git 状态

- 分支：`dev`
- 远程：`origin` → `https://github.com/xxz212/rag-assistant.git`
- 最近几个 commit：
  - `1c22768` fix: remove google-genai pin that conflicted with langchain-google-genai 4.2.2
  - `7b63fcc` fix: correct protobuf version and remove redundant google-generativeai pin
  - `53375ef` 之前的某次提交

---

## 8. 待完成事项（TO-DO）

### 立即要做（阻塞性）

- [x] **在 Streamlit Cloud 设置 `GOOGLE_API_KEY`** — 已完成
- [x] **确认 Streamlit Cloud 部署成功** — 已部署，URL：`https://rag-assistant-qfhteykrenpwycenjkaxpa.streamlit.app`

### 部署成功后

- [x] **更新 README 中的 live demo 链接**
  - URL：`https://rag-assistant-qfhteykrenpwycenjkaxpa.streamlit.app`

### 低优先级（可选，下次再做）

- [ ] 编写基础单元测试（`tests/` 目录，pytest）
- [ ] GitHub Actions CI（需要先有测试）

---

## 9. 重要约束（必读）

**财务约束**：用户预算极其有限，不允许在未经明确许可的情况下添加任何付费服务或订阅。当前使用的所有服务均为免费：
- Google Gemini 1.5 Flash：免费额度
- Streamlit Cloud：免费
- GitHub：免费
- ChromaDB：本地，免费

**代码原则**：
- 非必要不动代码，不影响运行逻辑
- 不擅自加功能、重构、引入新的依赖
- 有付费风险的操作必须先征得用户同意

---

## 10. 本地运行方式

```powershell
# 在项目根目录
.venv\Scripts\activate
streamlit run app.py
```

确保 `.env` 文件中有 `GOOGLE_API_KEY="你的key"`。
