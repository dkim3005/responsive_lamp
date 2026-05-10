from __future__ import annotations

import os
import re
import time

from config import HISTORY_MAX_MESSAGES, OPENAI_MODEL
from memory.store import MemoryStore

try:
    from dotenv import load_dotenv
    from openai import OpenAI
except Exception:  # pragma: no cover - optional runtime dependency
    load_dotenv = None
    OpenAI = None


SYSTEM_PROMPT = """You are a small, curious desk lamp with a warm friendly personality.
You can see what is happening around you and remember objects you have noticed.
Reply in 1 to 2 short sentences only.
When the user asks about an object you might have seen, always use query_memory.
If memory returns nothing, say honestly that you have not noticed it."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_memory",
            "description": "Search the lamp's visual memory for an object it has seen.",
            "parameters": {
                "type": "object",
                "properties": {"object_name": {"type": "string"}},
                "required": ["object_name"],
            },
        },
    }
]


class Agent:
    def __init__(self, store: MemoryStore) -> None:
        if load_dotenv is not None:
            load_dotenv()
        self.store = store
        self.client = OpenAI() if OpenAI is not None and os.getenv("OPENAI_API_KEY") else None
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_text: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        if self.client is None:
            return self._fallback_reply(user_text), (time.perf_counter() - t0) * 1000

        try:
            messages = self.history + [{"role": "user", "content": user_text}]
            first = self.client.chat.completions.create(model=OPENAI_MODEL, messages=messages, tools=TOOLS)
            msg = first.choices[0].message
            if msg.tool_calls:
                messages.append(msg)
                for call in msg.tool_calls:
                    if call.function.name == "query_memory":
                        object_name = _extract_json_arg(call.function.arguments, "object_name")
                        result = self._format_memory(object_name)
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                second = self.client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
                reply = second.choices[0].message.content or "I am not sure what I saw."
            else:
                reply = msg.content or self._fallback_reply(user_text)

            self.history.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}])
            self.history = [self.history[0]] + self.history[-HISTORY_MAX_MESSAGES:]
            return reply, (time.perf_counter() - t0) * 1000
        except Exception as exc:
            return f"{self._fallback_reply(user_text)} (LLM fallback: {type(exc).__name__})", (
                time.perf_counter() - t0
            ) * 1000

    def _fallback_reply(self, user_text: str) -> str:
        name = _guess_object_name(user_text)
        if name:
            return self._format_memory(name, natural=True)
        return "I am here and watching. Ask me where I last saw an object."

    def _format_memory(self, object_name: str, natural: bool = False) -> str:
        rows = self.store.query(object_name, limit=1)
        if not rows:
            return f"I have not noticed a {object_name} yet." if natural else "No record."
        row = rows[0]
        age = max(0, int(time.time() - row["last_seen"]))
        if natural:
            return f"I last saw the {row['label']} in the {row['zone']} about {age} seconds ago."
        return f"Found: {row['label']} at {row['zone']} (last seen {age}s ago, {row['count']} frames)."


def _guess_object_name(text: str) -> str:
    lowered = text.lower()
    match = re.search(r"(?:where(?: is|'s| did you see)?|find|seen|saw)\s+(?:my|the|a|an)?\s*([a-z0-9_-]+)", lowered)
    if match:
        return match.group(1).strip()
    for word in ["cup", "mug", "bottle", "book", "laptop", "phone", "keyboard", "mouse", "chair"]:
        if word in lowered:
            return "cup" if word == "mug" else word
    return ""


def _extract_json_arg(raw: str, key: str) -> str:
    match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', raw or "")
    return match.group(1) if match else ""

