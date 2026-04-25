import streamlit as st
import tempfile, os
from rag import build_qa, ask, generate_quiz

# ── 页面设置 ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Knowledge Assistant", layout="wide")

# ── 极光紫渐变风格 CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #1a0533 0%, #0a0a0f 50%, #001a33 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: rgba(20, 5, 40, 0.9) !important;
    border-right: 1px solid rgba(150, 80, 255, 0.3);
}
[data-testid="stSidebar"] h1 { color: #c084fc !important; }
h1 { color: #e9d5ff !important; }
.stCaption { color: #a78bfa !important; }
[data-testid="stChatInput"] {
    background: rgba(88, 28, 135, 0.2) !important;
    border: 1px solid rgba(167, 139, 250, 0.4) !important;
    border-radius: 12px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(88, 28, 135, 0.25) !important;
    border: 1px solid rgba(167, 139, 250, 0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(15, 5, 30, 0.6) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: 12px !important;
}
.stButton button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #9333ea, #6366f1) !important;
}
p, li, span, label, .uploadedFileName { color: #e2d9f3 !important; }
[data-testid="stFileUploaderDropzone"] * { color: #1a1a2e !important; }
[data-testid="stToolbar"] * { color: #1a1a2e !important; }
header[data-testid="stHeader"] { background: rgba(10, 5, 20, 0.95) !important; }
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
if "level" not in st.session_state:
    st.session_state.level = None  # 未选目标
if "show_level_select" not in st.session_state:
    st.session_state.show_level_select = False  # 是否显示选择按钮

# 等级对应的显示文字
LEVEL_LABELS = {
    "passed": "📗 Pass — 简洁直接",
    "credit": "📘 Credit — 清晰有条理",
    "distinction": "📙 Distinction — 详细全面",
    "high_distinction": "📕 High Distinction — 深度扩展",
}

# ── 侧边栏 ─────────────────────────────────────────────────────────────────────
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

            retriever, llm = build_qa([(p, s) for p, s, _ in files])
            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.session_state.name_map = {p: name for p, _, name in files}
            st.session_state.history = []
            st.session_state.level = None
            st.session_state.show_level_select = True  # 处理完显示选目标
        st.success(f"✅ {len(uploaded)} 个文档已就绪")

        # 已加载文档列表
    if "name_map" in st.session_state:
        st.markdown("---")
        st.markdown("**已加载文档：**")
        for name in st.session_state.name_map.values():
            st.caption(f"📄 {name}")

    # 显示当前目标等级
    if st.session_state.level:
        st.markdown("---")
        st.markdown(f"**当前目标：** {LEVEL_LABELS[st.session_state.level]}")

    # 清空对话
    if st.session_state.history:
        st.markdown("---")
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ── 主区域 ─────────────────────────────────────────────────────────────────────
st.title("✨ RAG Knowledge Assistant")
st.caption("上传文档，基于你的内容提问，答案100%来自文档。")
st.markdown("---")

if "retriever" not in st.session_state:
    st.info("👈 请先在左侧上传文档，然后开始提问。")

else:
    # ── 选择目标等级（处理完文档后显示一次）────────────────────────────────
    if st.session_state.show_level_select and not st.session_state.level:
        st.markdown("### 🎯 请选择你的目标成绩")
        st.caption("系统将根据你的目标调整回答的深度和风格")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📗 Pass\n简洁直接", use_container_width=True):
                st.session_state.level = "passed"
                st.session_state.show_level_select = False
                st.rerun()
        with col2:
            if st.button("📘 Credit\n清晰有条理", use_container_width=True):
                st.session_state.level = "credit"
                st.session_state.show_level_select = False
                st.rerun()
        with col3:
            if st.button("📙 Distinction\n详细全面", use_container_width=True):
                st.session_state.level = "distinction"
                st.session_state.show_level_select = False
                st.rerun()
        with col4:
            if st.button("📕 High Distinction\n深度扩展", use_container_width=True):
                st.session_state.level = "high_distinction"
                st.session_state.show_level_select = False
                st.rerun()

    # ── 已选目标，显示问答区 ──────────────────────────────────────────────
    elif st.session_state.level:
        # 出题按钮
        col_quiz, col_blank = st.columns([1, 4])
        with col_quiz:
            if st.button("🎯 出3道例题", use_container_width=True):
                with st.spinner("正在出题..."):
                    quiz = generate_quiz(
                        st.session_state.retriever,
                        st.session_state.llm,
                        st.session_state.level
                    )
                st.session_state.history.append({
                    "question": "🎯 生成例题",
                    "answer": quiz,
                    "sources": []
                })
                st.rerun()

        # 对话历史
        for item in st.session_state.history:
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.markdown(item["answer"])
                if item["sources"]:
                    st.caption("📄 来源：" + "、".join(item["sources"]))

        # 问题输入框
        question = st.chat_input("请输入你的问题...")
        if question:
            with st.spinner("思考中..."):
                answer, sources = ask(
                    st.session_state.retriever,
                    st.session_state.llm,
                    question,
                    st.session_state.level
                )
            name_map = st.session_state.get("name_map", {})
            display_sources = list({name_map.get(s, os.path.basename(s)) for s in sources})
            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "sources": display_sources
            })
            st.rerun()