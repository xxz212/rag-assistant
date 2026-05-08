import streamlit as st
import tempfile, os
from session import create_session, add_message, TYPE_DOC, TYPE_IMG, TYPE_DATA
from rag import build_qa, ask, generate_quiz
from vision import analyze_image, analyze_image_with_context, analyze_image_by_mode
from data_analysis import load_data, ask_data, generate_chart

# ── 页面设置 ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Knowledge Assistant", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
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
p, li, span, label { color: #e2d9f3 !important; }
[data-testid="stFileUploaderDropzone"] * { color: #000000 !important; }
[data-testid="stFileUploaderDropzone"] svg { fill: #333333 !important; stroke: #333333 !important; }
[data-testid="stFileUploaderDropzone"] button { background: rgba(0,0,0,0.07) !important; border: 1px solid rgba(0,0,0,0.15) !important; color: #8B4513 !important; }
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploaderDropzone"] button p { color: #8B4513 !important; }
[data-testid="stToolbar"] * { color: #1a1a2e !important; }
[data-testid="stMarkdownContainer"] p { color: #ffffff !important; }
header[data-testid="stHeader"] { background: rgba(10, 5, 20, 0.95) !important; }
.stAlert {
    background: rgba(88, 28, 135, 0.2) !important;
    border: 1px solid rgba(167, 139, 250, 0.3) !important;
}

/* 三点菜单按钮：未选中状态可见 */
[data-testid="stPopover"] > button,
[data-testid="stPopover"] button:first-child {
    background: rgba(88, 28, 135, 0.35) !important;
    color: #c084fc !important;
    border: 1px solid rgba(167, 139, 250, 0.4) !important;
    border-radius: 6px !important;
}
[data-testid="stPopover"] > button:hover,
[data-testid="stPopover"] button:first-child:hover {
    background: rgba(124, 58, 237, 0.6) !important;
    color: #ffffff !important;
}

/* 弹窗内文字对比度：弹窗背景为白色，强制深色文字 */
[data-testid="stPopoverBody"] p,
[data-testid="stPopoverBody"] span,
[data-testid="stPopoverBody"] label,
[data-testid="stPopoverBody"] li {
    color: #1a1a2e !important;
}
</style>
""", unsafe_allow_html=True)

# ── 初始化全局 session 列表 ────────────────────────────────────────────────────
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "active_id" not in st.session_state:
    st.session_state.active_id = None
if "pending_type" not in st.session_state:
    st.session_state.pending_type = None  # 记录待创建的session类型

LEVEL_LABELS = {
    "passed": "📗 Pass",
    "credit": "📘 Credit",
    "distinction": "📙 Distinction",
    "high_distinction": "📕 High Distinction",
}

def get_active():
    """获取当前激活的session"""
    for s in st.session_state.sessions:
        if s["id"] == st.session_state.active_id:
            return s
    return None

# ── 侧边栏 ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💬 Conversation List")

    # 新建对话按钮
    st.markdown("**Create a new conversation：**")
    # 点击按钮只记录类型，不立刻创建
    # 技术：把待创建类型存进session_state，触发主界面显示输入框
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 doc.", use_container_width=True):
          st.session_state.pending_type = TYPE_DOC
          st.rerun()
    with col2:
        if st.button("🖼️ image", use_container_width=True):
          st.session_state.pending_type = TYPE_IMG
          st.rerun()
    with col3:
        if st.button("📊 data", use_container_width=True):
          st.session_state.pending_type = TYPE_DATA
          st.rerun()

    st.markdown("---")

    # 颜色标记对照表
    COLOR_MAP = {
      "default": "",
      "red": "🔴",
      "yellow": "🟡",
      "green": "🟢",
      "blue": "🔵",
    }

    # 置顶的排前面，其余按创建顺序倒序
    # 技术：sorted()按pinned字段排序，pinned=True排最前
    sorted_sessions = sorted(
      st.session_state.sessions,
      key=lambda x: (not x.get("pinned", False),
                   st.session_state.sessions.index(x) * -1)
    )

    # 初始化重命名状态
    if "renaming_id" not in st.session_state:
        st.session_state.renaming_id = None

    for s in sorted_sessions:
        is_active = s["id"] == st.session_state.active_id
        color_icon = COLOR_MAP.get(s.get("color", "default"), "")
        pin_icon = "📌 " if s.get("pinned") else ""
        label = f"{'▶ ' if is_active else ''}{pin_icon}{color_icon} {s['name']}"

        # 每个对话框：左边是名字按钮，右边是三点菜单
        col_name, col_menu = st.columns([5, 1])

        with col_name:
            if st.button(label, use_container_width=True, key=f"btn_{s['id']}"):
               st.session_state.active_id = s["id"]
               st.session_state.renaming_id = None
               st.rerun()
        
        with col_menu:
          # 技术：st.popover 点击弹出操作面板
          with st.popover("⋯", use_container_width=True):

              # 重命名
              new_name = st.text_input(
                  "Rename",
                  value=s["name"],
                  key=f"rename_{s['id']}"
              )
              if st.button("✅ Confirm Rename", key=f"confirm_rename_{s['id']}", use_container_width=True):
                 s["name"] = new_name.strip() or s["name"]
                 st.rerun()

              st.divider()

              # 置顶/取消置顶
              pin_label = "📌 Unpin" if s.get("pinned") else "📌 Pin to Top"
              if st.button(pin_label, key=f"pin_{s['id']}", use_container_width=True):
                 s["pinned"] = not s.get("pinned", False)
                 st.rerun()

              st.divider()

              # 颜色标记
              st.markdown("🎨 <span style='color:#1a1a2e !important; font-weight:600'>Color Label</span>", unsafe_allow_html=True)
              color_cols = st.columns(5)
              for col, (color_key, color_emoji) in zip(
                  color_cols,
                  [("default","⬜"),("red","🔴"),("yellow","🟡"),("green","🟢"),("blue","🔵")]
              ):
                  with col:
                      if st.button(color_emoji, key=f"color_{s['id']}_{color_key}"):
                         s["color"] = color_key
                         st.rerun()

              st.divider()

              # 删除
              if st.button("🗑️ Delete", key=f"del_{s['id']}", use_container_width=True):
                  st.session_state.sessions = [
                      x for x in st.session_state.sessions
                      if x["id"] != s["id"]
                  ]
                  if st.session_state.active_id == s["id"]:
                      st.session_state.active_id = (
                          st.session_state.sessions[-1]["id"]
                          if st.session_state.sessions else None
                      )
                  st.rerun()


    

# ── 主区域 ─────────────────────────────────────────────────────────────────────
st.title("✨ RAG Knowledge Assistant")
st.markdown("---")

session = get_active()

# ── 自定义命名弹窗 ─────────────────────────────────────────────────────────
# 技术：检测pending_type，显示输入框让用户命名，确认后才真正创建session
if st.session_state.pending_type:
    type_names = {TYPE_DOC: "Document Chat", TYPE_IMG: "Image Analysis", TYPE_DATA: "Data Analysis"}
    type_icons = {TYPE_DOC: "📄", TYPE_IMG: "🖼️", TYPE_DATA: "📊"}
    ptype = st.session_state.pending_type

    st.markdown(f"### {type_icons[ptype]} New {type_names[ptype]}")
    name_input = st.text_input("Name this conversation:", placeholder=type_names[ptype])
    col_ok, col_cancel, _ = st.columns([1, 1, 4])
    with col_ok:
        if st.button("✅ Confirm", use_container_width=True):
            final_name = name_input.strip() or type_names[ptype]
            s = create_session(final_name, ptype)
            st.session_state.sessions.append(s)
            st.session_state.active_id = s["id"]
            st.session_state.pending_type = None
            st.rerun()
    with col_cancel:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.pending_type = None
            st.rerun()

elif session is None:
    st.info("👈 Create a new conversation from the sidebar to get started.")
# ══════════════════════════════════════════════════════════════════════════════
# 文档对话
# ══════════════════════════════════════════════════════════════════════════════
elif session["type"] == TYPE_DOC:
    st.caption(f"📄 Document Q&A | ID: {session['id']}")

    # 上传文件
    if not session["retriever"]:
        st.markdown("<span style='color:#8B4513 !important'>upload</span> file（PDF / TXT / DOCX）", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "upload file（PDF / TXT / DOCX）",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key=f"upload_doc_{session['id']}",
            label_visibility="collapsed"
        )
        if uploaded and st.button("⚡ handle file", key=f"process_{session['id']}"):
            with st.spinner("processing..."):
                files = []
                for f in uploaded:
                    suffix = f.name.split(".")[-1]
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}")
                    tmp.write(f.read())
                    tmp.close()
                    files.append((tmp.name, suffix, f.name))
                retriever, llm = build_qa([(p, s) for p, s, _ in files])
                session["retriever"] = retriever
                session["llm"] = llm
                session["name_map"] = {p: name for p, _, name in files}
                session["files"] = [name for _, _, name in files]
                session["show_level_select"] = True
                session["name"] = files[0][2]  # 用文件名作为对话标题
            st.rerun()

    # 选择目标等级
    elif session["show_level_select"] and not session["level"]:
        st.markdown("### 🎯 please select your goal score")
        col1, col2, col3, col4 = st.columns(4)
        levels = ["passed", "credit", "distinction", "high_distinction"]
        labels = ["📗 Pass\nbrief", "📘 Credit\nclear and slight logic",
                  "📙 Distinction\nDetailed and comprehensive", "📕 HD\ndeep expansive"]
        for col, level, label in zip([col1, col2, col3, col4], levels, labels):
            with col:
                if st.button(label, use_container_width=True, key=f"level_{level}_{session['id']}"):
                    session["level"] = level
                    session["show_level_select"] = False
                    st.rerun()

    # 问答区
    else:
        # 技术：用st.columns把页面分左右两栏
        # 左栏(70%)：正常聊天  右栏(30%)：出题专区
        chat_col, quiz_col = st.columns([7, 3])

        # ── 左栏：聊天区 ──────────────────────────────────────────────────────────
        with chat_col:
           st.caption(f"📄 {' | '.join(session['files'])}  |  Target: {LEVEL_LABELS.get(session['level'], '')}")

           # 对话历史（只显示普通问答，不显示出题）
           for item in session["history"]:
              if item.get("is_quiz"):
                 continue  # 出题内容跳过，不在聊天区显示
              with st.chat_message(item["role"]):
                 st.markdown(item["content"])
                 if item.get("sources"):
                    st.caption("📄 Sources: " + ", ".join(item["sources"]))

           # 输入框
           question = st.chat_input("Ask your question...", key=f"input_{session['id']}")
           if question:
              with st.spinner("Thinking..."):
                answer, sources = ask(session["retriever"], session["llm"], question, session["level"])
              name_map = session.get("name_map", {})
              display_sources = list({name_map.get(s, os.path.basename(s)) for s in sources})
              add_message(session, "user", question)
              add_message(session, "assistant", answer, display_sources)
              st.rerun()

        # ── 右栏：出题专区 ────────────────────────────────────────────────────────
        with quiz_col:
            st.markdown("### 🎯 Practice Questions")

            if st.button("Generate 3 Practice Questions", use_container_width=True, key=f"quiz_{session['id']}"):
                with st.spinner("Generating questions..."):
                  quiz = generate_quiz(session["retriever"], session["llm"], session["level"])
                # 存进session，标记is_quiz=True
                session["history"].append({
                    "role": "assistant",
                    "content": quiz,
                    "sources": [],
                    "is_quiz": True
                })
                st.rerun()

            # 显示最新一次出题结果
            import re
            quiz_items = [item for item in session["history"] if item.get("is_quiz")]
            if quiz_items:
                latest_quiz = quiz_items[-1]["content"]
                questions = re.split(r'\*\*题目\d+[：:]\*\*', latest_quiz)
                titles = re.findall(r'\*\*题目\d+[：:]\*\*', latest_quiz)
                for i, (title, block) in enumerate(zip(titles, questions[1:]), 1):
                   if "参考答案" in block:
                     parts = re.split(r'>?\s*💡\s*参考答案[：:]?', block)
                     question_text = parts[0].strip()
                     answer_text = parts[1].strip() if len(parts) > 1 else ""
                   else:
                     question_text = block.strip()
                     answer_text = ""
                   st.markdown(f"**Question {i}:**")
                   st.markdown(question_text)
                   if answer_text:
                       with st.expander(f"👀 Reveal Answer"):
                         st.markdown(f"💡 {answer_text}")
                   st.divider()
            else:
               st.caption("Click the button above to generate practice questions")

# ══════════════════════════════════════════════════════════════════════════════
# 图片对话
# ══════════════════════════════════════════════════════════════════════════════
elif session["type"] == TYPE_IMG:
    st.caption(f"🖼️ image analysis | ID: {session['id']}")

    MODE_LABELS = {
        "ecommerce": "🛒 E-commerce Product",
        "contract":  "📄 Contract / Invoice",
        "competitor":"🔍 Competitor Analysis",
        "interior":  "🏠 Interior Design",
        "medical":   "🏥 Medical Report Review",
    }

    # ── Step 1: Select analysis mode ─────────────────────────────────────────
    if not session.get("mode"):
        st.markdown("### 🎯 Select Analysis Mode")
        c1, c2, c3 = st.columns(3)
        c4, c5, _ = st.columns(3)
        with c1:
            if st.button("🛒 E-commerce Product", use_container_width=True, key=f"mode_ecom_{session['id']}"):
                session["mode"] = "ecommerce"
                st.rerun()
        with c2:
            if st.button("📄 Contract / Invoice", use_container_width=True, key=f"mode_contract_{session['id']}"):
                session["mode"] = "contract"
                st.rerun()
        with c3:
            if st.button("🔍 Competitor Analysis", use_container_width=True, key=f"mode_comp_{session['id']}"):
                session["mode"] = "competitor"
                st.rerun()
        with c4:
            if st.button("🏠 Interior Design", use_container_width=True, key=f"mode_interior_{session['id']}"):
                session["mode"] = "interior"
                st.rerun()
        with c5:
            if st.button("🏥 Medical Report Review", use_container_width=True, key=f"mode_medical_{session['id']}"):
                session["mode"] = "medical"
                st.rerun()

    # ── 第二步：上传图片 & 分析 ───────────────────────────────────────────────
    else:
        col_mode, col_reset = st.columns([5, 1])
        with col_mode:
            st.markdown(f"**Current Mode: {MODE_LABELS[session['mode']]}**")
        with col_reset:
            if st.button("Switch Mode", key=f"reset_mode_{session['id']}", use_container_width=True):
                session["mode"] = None
                session["history"] = []
                session["current_image"] = None
                session["medical_confirmed"] = False
                st.rerun()

        # 医疗模式：免责声明前置确认
        if session["mode"] == "medical" and not session.get("medical_confirmed"):
            st.warning("⚠️ **Disclaimer**: This feature is for reference only. AI analysis does not constitute a medical diagnosis and cannot replace the advice of a licensed physician. Please consult a qualified healthcare professional for any medical concerns.")
            if st.button("✅ I understand, proceed", key=f"medical_confirm_{session['id']}"):
                session["medical_confirmed"] = True
                st.rerun()
        else:
            st.markdown("<span style='color:#8B4513 !important'>upload</span> image（JPG / PNG）", unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "upload image（JPG / PNG）",
                type=["jpg", "jpeg", "png"],
                key=f"upload_img_{session['id']}",
                label_visibility="collapsed"
            )

            if uploaded and session.get("name") != uploaded.name:
                suffix = uploaded.name.split(".")[-1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}")
                tmp.write(uploaded.read())
                tmp.close()
                session["current_image"] = tmp.name
                session["name"] = uploaded.name
                session["history"] = []
                st.rerun()

            if session.get("current_image"):
                st.image(session["current_image"], caption=session.get("name", ""), use_column_width=True)

                # 自动分析（首次，历史为空时触发）
                if not session["history"]:
                    with st.spinner("Analyzing..."):
                        result = analyze_image_by_mode(session["current_image"], session["mode"])
                    add_message(session, "assistant", result)
                    st.rerun()

                for item in session["history"]:
                    with st.chat_message(item["role"]):
                        st.markdown(item["content"])

                question = st.chat_input("Ask a follow-up question...", key=f"input_{session['id']}")
                if question:
                    with st.spinner("Analyzing..."):
                        answer = analyze_image_by_mode(session["current_image"], session["mode"], followup=question)
                    add_message(session, "user", question)
                    add_message(session, "assistant", answer)
                    st.rerun()
            else:
                st.info("👆 Upload an image to start analysis")

# ══════════════════════════════════════════════════════════════════════════════
# 数据对话
# ══════════════════════════════════════════════════════════════════════════════
elif session["type"] == TYPE_DATA:
    st.caption(f"📊 data analysis | ID: {session['id']}")

    st.markdown("<span style='color:#8B4513 !important'>upload</span> data file（CSV / Excel）", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "upload data file（CSV / Excel）",
        type=["csv", "xlsx", "xls"],
        key=f"upload_data_{session['id']}",
        label_visibility="collapsed"
    )

    if uploaded:
        suffix = uploaded.name.split(".")[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}")
        tmp.write(uploaded.read())
        tmp.close()
        import pandas as pd
        df = load_data(tmp.name, suffix)
        session["df"] = df
        session["name"] = uploaded.name
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

    # 对话历史
    for item in session["history"]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            # 如果有图表路径，显示下载按钮
            if item.get("chart_path") and os.path.exists(item["chart_path"]):
                with open(item["chart_path"], "rb") as f:
                    st.download_button(
                        "📥 Download Chart",
                        f,
                        file_name="chart.png",
                        mime="image/png",
                        key=f"dl_{item['chart_path']}"
                    )

    # 输入框
    if session.get("df") is not None:
        question = st.chat_input(
            "Ask a question about the data, or type 'generate chart: ...' to create a chart",
            key=f"input_{session['id']}"
        )
        if question:
            add_message(session, "user", question)
            if question.lower().startswith("generate chart:"):
                with st.spinner("Generating chart..."):
                    chart_path = tempfile.mktemp(suffix=".png")
                    generate_chart(session["df"], question, chart_path)
                msg = {"role": "assistant", "content": "Chart generated — click below to download:",
                       "sources": [], "chart_path": chart_path}
                session["history"].append(msg)
            else:
                with st.spinner("analyzing..."):
                    answer = ask_data(session["df"], question)
                add_message(session, "assistant", answer)
            st.rerun()
    else:
        st.info("👆 please upload data file here")