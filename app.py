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
    background: radial-gradient(ellipse at 20% 50%, #1a0533 0%, #12082a 50%, #001a33 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: rgba(20, 5, 40, 0.9) !important;
    border-right: 1px solid rgba(150, 80, 255, 0.3) !important;
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
[data-testid="stExpandSidebarButton"] {
    background: rgba(124, 58, 237, 0.6) !important;
    border-radius: 8px !important;
}
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] * {
    color: #ffffff !important;
}
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
# ── 初始化首页状态 ─────────────────────────────────────────────────────────────
if "show_landing" not in st.session_state:
    st.session_state.show_landing = True
if "landing_card" not in st.session_state:
    st.session_state.landing_card = 0
# ── 初始化全局 session 列表 ────────────────────────────────────────────────────
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "active_id" not in st.session_state:
    st.session_state.active_id = None
if "pending_type" not in st.session_state:
    st.session_state.pending_type = None

LEVEL_LABELS = {
    "passed": "📗 Pass",
    "credit": "📘 Credit",
    "distinction": "📙 Distinction",
    "high_distinction": "📕 High Distinction",
}

def get_active():
    for s in st.session_state.sessions:
        if s["id"] == st.session_state.active_id:
            return s
    return None

# ── 侧边栏（只在非首页时渲染，避免 CSS display hack 导致折叠状态异常）─────────
if not st.session_state.show_landing:
    with st.sidebar:
        st.title("💬 Conversation List")

        # 新建对话按钮
        st.markdown("**Create a new conversation：**")
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

            col_name, col_menu = st.columns([5, 1])

            with col_name:
                if st.button(label, use_container_width=True, key=f"btn_{s['id']}"):
                   st.session_state.active_id = s["id"]
                   st.session_state.renaming_id = None
                   st.rerun()

            with col_menu:
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

        # 返回首页
        st.markdown("---")
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.show_landing = True
            st.rerun()


# ── 主区域 ─────────────────────────────────────────────────────────────────────
st.markdown("---")

# ── 首页轮播 ───────────────────────────────────────────────────────────────────
if st.session_state.show_landing:

    # 首页额外CSS：毛玻璃卡片 + 女性化渐变
    st.markdown("""
    <style>
    /* 收起状态下的箭头按钮 */
    [data-testid="collapsedControl"] svg {
       fill: #c084fc !important;
       color: #c084fc !important;
    }
    button[kind="header"] {
       color: #c084fc !important;
    }

    .landing-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f9a8d4, #c084fc, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .landing-subtitle {
        text-align: center;
        color: #c4b5fd;
        font-size: 1rem;
        margin-bottom: 2.5rem;
        letter-spacing: 0.05em;
    }
    .card-container {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(244, 168, 212, 0.25);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
        min-height: 380px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .card-icon { font-size: 4rem; margin-bottom: 1rem; }
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f9a8d4;
        margin-bottom: 0.8rem;
    }
    .card-desc {
        color: #c4b5fd;
        font-size: 0.9rem;
        line-height: 1.7;
        margin-bottom: 1.5rem;
    }
    .card-tags {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: center;
        margin-bottom: 1.5rem;
    }
    .tag {
        background: rgba(244, 168, 212, 0.15);
        border: 1px solid rgba(244, 168, 212, 0.3);
        color: #f9a8d4;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
    }
    .dot-container {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin-top: 1.5rem;
    }
    .dot { font-size: 0.6rem; }
    </style>
    """, unsafe_allow_html=True)

    # 标题
    st.markdown('<div class="landing-title">✨ RAG Knowledge Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-subtitle">Your intelligent document companion — powered by AI</div>', unsafe_allow_html=True)

    # 卡片数据
    cards = [
        {
            "icon": "📄",
            "title": "Document Q&A",
            "desc": "Upload your PDF, TXT or DOCX files and ask anything.\nGet grounded answers with source references — no hallucination.",
            "tags": ["PDF", "TXT", "DOCX", "RAG", "Source Citation"],
            "type": TYPE_DOC,
            "gradient": "linear-gradient(135deg, rgba(192,132,252,0.15), rgba(244,168,212,0.08))"
        },
        {
            "icon": "🖼️",
            "title": "Image Analysis",
            "desc": "Upload product photos, contracts, or competitor screenshots.\nAI describes, analyzes, and answers your questions visually.",
            "tags": ["Product", "Contract", "Competitor", "Visual Q&A"],
            "type": TYPE_IMG,
            "gradient": "linear-gradient(135deg, rgba(244,168,212,0.15), rgba(251,207,232,0.08))"
        },
        {
            "icon": "📊",
            "title": "Data Analysis",
            "desc": "Upload CSV or Excel files.\nAsk data questions in plain language and generate downloadable charts.",
            "tags": ["CSV", "Excel", "Charts", "Statistics"],
            "type": TYPE_DATA,
            "gradient": "linear-gradient(135deg, rgba(129,140,248,0.15), rgba(192,132,252,0.08))"
        },
    ]

    current = st.session_state.landing_card

    # 左右切换 + 卡片显示
    col_left, col_card, col_right = st.columns([1, 6, 1])

    with col_left:
        st.markdown("<div style='height:160px'></div>", unsafe_allow_html=True)
        if st.button("‹", use_container_width=True, key="prev_card"):
            st.session_state.landing_card = (current - 1) % len(cards)
            st.rerun()

    with col_card:
        card = cards[current]
        tags_html = "".join([f'<span class="tag">{t}</span>' for t in card["tags"]])
        st.markdown(f"""
        <div class="card-container" style="background:{card['gradient']}">
            <div class="card-icon">{card['icon']}</div>
            <div class="card-title">{card['title']}</div>
            <div class="card-desc">{card['desc']}</div>
            <div class="card-tags">{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # 指示点
        dots = "".join([
            f'<span class="dot">{"🌸" if i == current else "○"}</span>'
            for i in range(len(cards))
        ])
        st.markdown(f'<div class="dot-container">{dots}</div>', unsafe_allow_html=True)

        # 进入按钮
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if st.button(f"Enter {card['title']} →", use_container_width=True, key="enter_card"):
            s = create_session(card['title'], card['type'])
            st.session_state.sessions.append(s)
            st.session_state.active_id = s["id"]
            st.session_state.show_landing = False
            st.rerun()

    with col_right:
        st.markdown("<div style='height:160px'></div>", unsafe_allow_html=True)
        if st.button("›", use_container_width=True, key="next_card"):
            st.session_state.landing_card = (current + 1) % len(cards)
            st.rerun()

    # 底部返回提示
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#6b7280;font-size:0.8rem'>← › 切换功能 · 点击 Enter 开始使用</div>", unsafe_allow_html=True)

    st.stop()  # 首页显示完就停止，不渲染后面的内容

session = get_active()

# ── 自定义命名弹窗 ─────────────────────────────────────────────────────────
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
                try:
                    retriever, llm = build_qa([(p, s) for p, s, _ in files])
                    session["retriever"] = retriever
                    session["llm"] = llm
                    session["name_map"] = {p: name for p, _, name in files}
                    session["files"] = [name for _, _, name in files]
                    session["show_level_select"] = True
                    session["name"] = files[0][2]
                except Exception as e:
                    st.error(f"⚠️ File processing failed: {e}")
                    st.stop()
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
        chat_col, quiz_col = st.columns([7, 3])

        # ── 左栏：聊天区 ──────────────────────────────────────────────────────────
        with chat_col:
           st.caption(f"📄 {' | '.join(session['files'])}  |  Target: {LEVEL_LABELS.get(session['level'], '')}")

           for item in session["history"]:
              if item.get("is_quiz"):
                 continue
              with st.chat_message(item["role"]):
                 st.markdown(item["content"])
                 if item.get("sources"):
                    st.caption("📄 Sources: " + ", ".join(item["sources"]))

           question = st.chat_input("Ask your question...", key=f"input_{session['id']}")
           if question:
              try:
                  with st.spinner("Thinking..."):
                      answer, sources = ask(session["retriever"], session["llm"], question, session["level"])
                  name_map = session.get("name_map", {})
                  display_sources = list({name_map.get(s, os.path.basename(s)) for s in sources})
                  add_message(session, "user", question)
                  add_message(session, "assistant", answer, display_sources)
              except Exception as e:
                  st.error(f"⚠️ Failed to get answer: {e}")
              st.rerun()

        # ── 右栏：出题专区 ────────────────────────────────────────────────────────
        with quiz_col:
            st.markdown("### 🎯 Practice Questions")

            if st.button("Generate 3 Practice Questions", use_container_width=True, key=f"quiz_{session['id']}"):
                try:
                    with st.spinner("Generating questions..."):
                        quiz = generate_quiz(session["retriever"], session["llm"], session["level"])
                    session["history"].append({
                        "role": "assistant",
                        "content": quiz,
                        "sources": [],
                        "is_quiz": True
                    })
                except Exception as e:
                    st.error(f"⚠️ Quiz generation failed: {e}")
                st.rerun()

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
                    try:
                        with st.spinner("Analyzing..."):
                            result = analyze_image_by_mode(session["current_image"], session["mode"])
                        add_message(session, "assistant", result)
                    except Exception as e:
                        st.error(f"⚠️ Image analysis failed: {e}")
                    st.rerun()

                for item in session["history"]:
                    with st.chat_message(item["role"]):
                        st.markdown(item["content"])

                question = st.chat_input("Ask a follow-up question...", key=f"input_{session['id']}")
                if question:
                    try:
                        with st.spinner("Analyzing..."):
                            answer = analyze_image_by_mode(session["current_image"], session["mode"], followup=question)
                        add_message(session, "user", question)
                        add_message(session, "assistant", answer)
                    except Exception as e:
                        st.error(f"⚠️ Analysis failed: {e}")
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
                try:
                    with st.spinner("Generating chart..."):
                        chart_path = tempfile.mktemp(suffix=".png")
                        generate_chart(session["df"], question, chart_path)
                    msg = {"role": "assistant", "content": "Chart generated — click below to download:",
                           "sources": [], "chart_path": chart_path}
                    session["history"].append(msg)
                except Exception as e:
                    st.error(f"⚠️ Chart generation failed: {e}")
            else:
                try:
                    with st.spinner("analyzing..."):
                        answer = ask_data(session["df"], question)
                    add_message(session, "assistant", answer)
                except Exception as e:
                    st.error(f"⚠️ Data analysis failed: {e}")
            st.rerun()
    else:
        st.info("👆 please upload data file here")
