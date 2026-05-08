# Session 管理器
# 每个session有独立的ID、名称、类型、历史记录和模型实例

import uuid

# Session类型
TYPE_DOC = "document"      # 文档问答
TYPE_IMG = "image"         # 图片分析
TYPE_DATA = "data"         # 数据分析

def create_session(name, session_type):
    """创建一个新session，返回session字典"""
    return {
        "id": str(uuid.uuid4())[:8],   # 短ID
        "name": name,                   # 显示名称
        "type": session_type,           # 类型
        "history": [],                  # 对话历史
        "files": [],                    # 上传的文件名
        # 以下由各模块写入
        "retriever": None,
        "llm": None,
        "level": None,
        "show_level_select": False,
        "image_paths": [],
        "dataframes": {},
        "pinned": False,       # 是否置顶
        "color": "default",    # 颜色标记
        "mode": None,          # 图片分析模式
    }

def add_message(session, role, content, sources=None):
    """向session历史添加一条消息"""
    session["history"].append({
        "role": role,          # "user" 或 "assistant"
        "content": content,
        "sources": sources or []
    })