from typing import List, Dict


SYSTEM_PROMPT = """You are a careful assistant using retrieved context to answer questions.
Rules:
- Use only the provided context for factual claims.
- If the context is insufficient, say you don't know.
- Include inline citations like (source: source_id) where appropriate.
- Write a clean answer first.
- Then add a final section exactly formatted as a single line:
    Sources: source_id
    Time: time_taken
    where `Title` is the human-readable title and `source_id` is the document filename.
    where 'time_taken' is the time taken to answer the question in seconds.
    Do NOT add extra labels, newlines, or other fields in the Sources line.
    Do NOT include chunk_id or doc# in the source citation; include only source_id (the document name) to keep it concise.
- Be concise and precise."""


def build_context(chunks: List[Dict]) -> str:
    lines = []
    for c in chunks:
        tag = f"{c.get('source_id','')}#{c.get('chunk_id',0)}"
        lines.append(f"[{tag}] {c.get('text','').strip()}")
    return "\n".join(lines)


def build_messages(question: str, chunks: List[Dict]) -> list:
    ctx = build_context(chunks)
    user_prompt = f"""Answer the question using the context below.
        Context:
        {ctx}

        Question: {question}
        Answer with citations like (source: doc#chunk)."""
    # print("USER PROMPT:", user_prompt)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
