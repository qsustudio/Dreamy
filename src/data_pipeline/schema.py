"""统一的数据格式。

有两种数据形态，清洗和打包时要区别对待：

1. **对话**（chat / empathy / depth / persona / safety）
   多轮 user/assistant 交替，打包时加角色标记。

2. **文档**（general 通用语料）
   整段纯文本（新闻/百科/小说），没有轮次概念。
   如果用对话的轮数规则去筛，会被全部误删 —— 这是从零训练最容易踩的坑。

磁盘格式（JSONL，每行一条）::

    {"turns": [{"role": "user", "text": "..."},
               {"role": "assistant", "text": "..."}],
     "meta": {"category": "chat"}}

    {"text": "整段文档正文", "meta": {"category": "general"}}

原始数据格式千奇百怪，由 load_raw_file() 负责适配常见几种。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

ROLES = ("user", "assistant")
DOCUMENT_ROLE = "document"


@dataclass
class Turn:
    role: str
    text: str

    def to_dict(self) -> dict:
        return {"role": self.role, "text": self.text}


@dataclass
class Conversation:
    turns: list[Turn]
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- 便捷属性 ----
    @property
    def is_document(self) -> bool:
        """纯文本文档（通用语料），不是对话。"""
        return len(self.turns) == 1 and self.turns[0].role == DOCUMENT_ROLE

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def total_chars(self) -> int:
        return sum(len(t.text) for t in self.turns)

    @property
    def text(self) -> str:
        """整条内容拼成一个字符串，用于去重/SimHash。"""
        return "\n".join(t.text for t in self.turns)

    def to_dict(self) -> dict:
        return {
            "turns": [t.to_dict() for t in self.turns],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        turns = [
            Turn(role=str(t.get("role", "")), text=str(t.get("text", "")))
            for t in d.get("turns", [])
        ]
        return cls(turns=turns, meta=dict(d.get("meta") or {}))


# ---------------------------------------------------------------- 适配器
def _from_turn_list(items: list, meta: dict) -> Conversation | None:
    """[{role, content}] / [{role, text}] / [{from, value}] 这类结构。"""
    turns = []
    for it in items:
        if not isinstance(it, dict):
            return None
        role = it.get("role") or it.get("from") or it.get("speaker")
        text = it.get("text") or it.get("content") or it.get("value")
        if role is None or text is None:
            return None
        role = str(role).lower()
        # 常见别名映射
        if role in ("human", "user", "客户", "用户", "human_user"):
            role = "user"
        elif role in ("gpt", "assistant", "bot", "ai", "系统", "助手", "回复"):
            role = "assistant"
        else:
            return None
        turns.append(Turn(role=role, text=str(text)))
    return Conversation(turns=turns, meta=meta) if turns else None


def _from_pair(d: dict, meta: dict) -> Conversation | None:
    """指令-回复对：question/answer、prompt/response、instruction/output 等。"""
    pairs = [
        ("instruction", "output"),
        ("prompt", "response"),
        ("question", "answer"),
        ("query", "reply"),
        ("user", "assistant"),
        ("问", "答"),
    ]
    for k_user, k_asst in pairs:
        if k_user in d and k_asst in d:
            u, a = d[k_user], d[k_asst]
            if isinstance(u, str) and isinstance(a, str) and u.strip() and a.strip():
                return Conversation(
                    turns=[Turn("user", u), Turn("assistant", a)], meta=meta
                )
    return None


def adapt_record(d: Any, meta: dict) -> Conversation | None:
    """把一条原始记录转成 Conversation；认不出来返回 None。"""
    if isinstance(d, str):
        # 纯字符串 = 一段文档（通用语料常见）
        return Conversation([Turn(DOCUMENT_ROLE, d)], meta) if d.strip() else None
    if isinstance(d, list):
        return _from_turn_list(d, meta)
    if not isinstance(d, dict):
        return None

    # 文档：{"text": ...} / {"content": ...} / {"document": ...}
    for key in ("text", "content", "document", "body", "article"):
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            return Conversation([Turn(DOCUMENT_ROLE, v)], meta)

    # 嵌套在常见字段里的对话
    for key in ("turns", "messages", "conversations", "dialogue",
                "dialog", "conversation", "chat"):
        v = d.get(key)
        if isinstance(v, list):
            conv = _from_turn_list(v, meta)
            if conv:
                return conv

    return _from_pair(d, meta)


# ---------------------------------------------------------------- 读写
def _load_records(path: str, ext: str) -> list:
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # 不是合法 JSON 就当一行文档用
                records.append(line)
    return records


def load_raw_file(
    path: str,
    meta: dict | None = None,
    document: bool = False,
) -> Iterator[Conversation]:
    """读取一个原始文件，逐条产出 Conversation。

    Args:
        document: True 表示这是通用语料（纯文本），
                  .txt 按「一行一段」解析，而不是找对话标记。
    """
    meta = dict(meta or {})
    ext = os.path.splitext(path)[1].lower()

    if ext in (".jsonl", ".json"):
        for rec in _load_records(path, ext):
            conv = adapt_record(rec, meta)
            if conv:
                yield conv

    elif ext in (".txt", ".text"):
        if document:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield Conversation([Turn(DOCUMENT_ROLE, line)], meta)
        else:
            conv = _parse_marked_text(path, meta)
            if conv:
                yield conv

    # 其它扩展名直接跳过（由调用方在报告里提示）


_CN_USER_MARKS = ("用户:", "用户：", "user:", "User:", "问:", "问：", "A:", "A：")
_CN_ASST_MARKS = ("助手:", "助手：", "assistant:", "Assistant:", "答:", "答：", "B:", "B：")


def _parse_marked_text(path: str, meta: dict) -> Conversation | None:
    """解析「用户: xxx / 助手: yyy」交替的纯文本。"""
    turns: list[Turn] = []
    cur_role, cur_buf = None, []

    def flush():
        if cur_role and cur_buf:
            text = "".join(cur_buf).strip()
            if text:
                turns.append(Turn(cur_role, text))

    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            matched = False
            for m in _CN_USER_MARKS:
                if line.startswith(m):
                    flush()
                    cur_role, cur_buf = "user", [line[len(m):].strip()]
                    matched = True
                    break
            if matched:
                continue
            for m in _CN_ASST_MARKS:
                if line.startswith(m):
                    flush()
                    cur_role, cur_buf = "assistant", [line[len(m):].strip()]
                    matched = True
                    break
            if matched:
                continue
            if cur_role:
                cur_buf.append(line)

    flush()
    return Conversation(turns=turns, meta=meta) if turns else None


def dump_jsonl(convs: list[Conversation], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[Conversation]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Conversation.from_dict(json.loads(line))
