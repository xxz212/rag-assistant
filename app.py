import streamlit as st
import tempfile, os
from rag import build_qa, ask

# ── 页面设置 ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Knowledge Assistant", layout="wide")

# ── 极光紫渐变风格 CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 背景：深黑+极光紫渐变光晕 */
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #1a0533 0%, #0a0a0f 50%, #001a33 100%);
    min-height: 100vh;
}

/* 侧边栏背景 */
[data-testid="stSidebar"] {
    background: rgba(20, 5, 40, 0.9) !important;
    border-right: 1px solid rgba(150, 80, 255, 0.3);
}

/* 顶部header背景 */
header[data-testid="stHeader"] {
    background: rgba(10, 5, 20, 0.95) !important;
}            
            
/* 侧边栏标题 */
[data-testid="stSidebar"] h1 {
    color: #c084fc !important;
}

/* 主标题 */
h1 { color: #e9d5ff !important; }

/* 副标题/caption */
.stCaption { color: #a78bfa !important; }

/* 聊天输入框 */
[data-testid="stChatInput"] {
    background: rgba(88, 28, 135, 0.2) !important;
    border: 1px solid rgba(167, 139, 250, 0.4) !important;
    border-radius: 12px !important;
}

/* 用户消息气泡 */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(88, 28, 135, 0.25) !important;
    border: 1px solid rgba(167, 139, 250, 0.3) !important;
    border-radius: 12px !important;
}

/* AI消息气泡 */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(15, 5, 30, 0.6) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: 12px !important;
}

/* 按钮 */
.stButton button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #9333ea, #6366f1) !important;
}

/* 文字颜色 */
p, li, span, label, .uploadedFileName { color: #e2d9f3 !important; }


/* Upload按钮和提示文字 */
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] * {
    color: #1a1a2e !important;
}

/* 右上角Deploy */
[data-testid="stToolbar"] *,
[data-testid="stToolbar"] span,
[data-testid="stToolbar"] a {
    color: #e2d9f3 !important;
}

/* info框 */
.stAlert { 
    background: rgba(88, 28, 135, 0.2) !important;
    border: 1px solid rgba(167, 139, 250, 0.3) !important;
    color: #c084fc !important;
}
</style>
""", unsafe_allow_html=True)

# ── 初始化 session 状态 ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── 侧边栏：上传 + 文件管理 ────────────────────────────────────────────────────
with st.sidebar:
    st.title("📂 文档管理")
    uploaded = st.file_uploader(
        "上传文档（PDF / TXT / DOCX）",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded and st.button("⚡ 处理文档", use_container_width=True):
        with st.spinner("正在处理..."):
            files = []
            for f in uploaded:
                suffix = f.name.split(".")[-1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}")
                tmp.write(f.read())
                tmp.close()
                files.append((tmp.name, suffix, f.name))

            chain, retriever = build_qa([(p, s) for p, s, _ in files])
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.session_state.name_map = {p: name for p, _, name in files}
            st.session_state.history = []
        st.success(f"✅ {len(uploaded)} 个文档已就绪")

    if "name_map" in st.session_state:
        st.markdown("---")
        st.markdown("**已加载文档：**")
        for name in st.session_state.name_map.values():
            st.caption(f"📄 {name}")

    if st.session_state.history:
        st.markdown("---")
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ── 主区域 ─────────────────────────────────────────────────────────────────────
st.title("✨ RAG Knowledge Assistant")
st.caption("上传文档，基于你的内容提问，答案100%来自文档。")
st.markdown("---")

if "chain" not in st.session_state:
    st.info("👈 请先在左侧上传文档，然后开始提问。")
else:
    question = st.chat_input("请输入你的问题...")

    if question:
        with st.spinner("思考中..."):
            answer, sources = ask(st.session_state.chain, st.session_state.retriever, question)
        name_map = st.session_state.get("name_map", {})
        display_sources = list({name_map.get(s, os.path.basename(s)) for s in sources})
        st.session_state.history.append({
            "question": question,
            "answer": answer,
            "sources": display_sources
        })

    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            if item["sources"]:
                st.caption("📄 来源：" + "、".join(item["sources"]))