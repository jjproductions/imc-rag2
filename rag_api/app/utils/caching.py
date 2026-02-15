import hashlib, re, unicodedata
import json
from typing import Any, Dict, List, Tuple, Optional


def make_cache_key(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    index_version: Optional[str] = None,  # bump this when your corpus changes
) -> str:
    """
    Create a stable key across the stream=True and stream=False calls for the same user turn.
    """
    payload = {
        "model": model,
        "messages": messages,  # must be deterministic order and content
        "temperature": temperature,
        "max_tokens": max_tokens,
        "index_version": index_version or "v1",
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    # print(f"Message: {json.dumps(messages, indent=2)}")
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonicalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_final_user_message(raw_input: str) -> str:
    """
    Handles both:
    1) Full instruction + <chat_history> blocks
    2) Plain user question strings
    """

    # Case 1: <chat_history> exists
    if "<chat_history>" in raw_input and "</chat_history>" in raw_input:
        chat_block = re.search(
            r"<chat_history>(.*?)</chat_history>",
            raw_input,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if chat_block:
            history = chat_block.group(1)

            # Find all USER messages
            user_messages = re.findall(
                r"USER:\s*(.*?)(?=\nASSISTANT:|\Z)",
                history,
                flags=re.DOTALL | re.IGNORECASE,
            )

            if user_messages:
                return canonicalize_text(user_messages[-1])

    # Case 2: No chat history → treat entire input as the user query
    return canonicalize_text(raw_input)


def make_retrieval_cache_key(
    question: str,
    top_k: int,
    index_version: Optional[str] = None,
) -> str:
    """Create a stable cache key for retrieval results."""
    payload = {
        "q": extract_final_user_message(question),
        "k": top_k,
        "index_version": index_version or "v1",
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
