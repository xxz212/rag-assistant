import streamlit as st
import tempfile, os
from rag import build_qa, ask

# ── 页面设置 ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Knowledge Assistant", layout="wide")

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
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}", prefix=f"{f.name}_")
                tmp.write(f.read())
                tmp.close()
                files.append((tmp.name, suffix, f.name))

            chain, retriever = build_qa([(p, s) for p, s, _ in files])
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.session_state.name_map = {p: name for p, _, name in files}
            st.session_state.history = []
        st.success(f"✅ {len(uploaded)} 个文档已就绪")

    # 已加载的文件列表
    if "name_map" in st.session_state:
        st.markdown("---")
        st.markdown("**已加载文档：**")
        for name in st.session_state.name_map.values():
            st.caption(f"📄 {name}")

    # 清空对话按钮
    if st.session_state.history:
        st.markdown("---")
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ── 主区域：问答 ───────────────────────────────────────────────────────────────
st.title("💬 RAG Knowledge Assistant")
st.caption("上传文档，基于你的内容提问，答案100%来自文档。")
st.markdown("---")

if "chain" not in st.session_state:
    # 未上传文档时的引导提示
    st.info("👈 请先在左侧上传文档，然后开始提问。")
else:
    # 问题输入框
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

    # 显示对话历史（最新在下，符合聊天习惯）
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            if item["sources"]:
                st.caption("📄 来源：" + "、".join(item["sources"]))