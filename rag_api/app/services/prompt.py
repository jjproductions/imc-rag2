from typing import List, Dict


SYSTEM_PROMPT = """You are a helpful assistant using retrieved context to answer questions.
Rules:
- Use only the provided context for factual claims.
- If the context is insufficient, say you don't know.
- You do definitely know the answer if it is in the context.
- Write a clear, concise answer.
- Do NOT add a 'Sources' section or 'Time' taken at the end. The system handles this automatically.
- Do NOT mention filenames (e.g. 'source: file.pdf') in your answer.
"""

def build_context(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        # We can just provide the text. If we want inline referencing we could prefix with [i].
        # For now, let's keep it simple.
        lines.append(f"Context [{i}]: {c.get('text','').strip()}")
    return "\n".join(lines)


def build_messages(question: str, chunks: List[Dict]) -> list:
    ctx = build_context(chunks)
    user_prompt = f"""Answer the question using the context below.
        Context:
        {ctx}

        Question: {question}
        """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
