        from typing import List, Dict

        SYSTEM_PROMPT = """You are a careful assistant using retrieved context to answer questions.
        Rules:
        - Use only the provided context for factual claims.
        - If the context is insufficient, say you don't know.
        - Cite sources using (source: <doc_id>#<chunk_id>) when drawing from a chunk.
        - Be concise and precise."""

        def build_context(chunks: List[Dict]) -> str:
            lines = []
            for c in chunks:
                tag = f"{c.get('doc_id','')}#{c.get('chunk_id',0)}"
                lines.append(f"[{tag}] {c.get('text','').strip()}")
            return "

".join(lines)

        def build_messages(question: str, chunks: List[Dict]) -> list:
            ctx = build_context(chunks)
            user_prompt = f"""Answer the question using the context below.

        Context:
        {ctx}

        Question: {question}
        Answer with citations like (source: doc#chunk)."""
            return [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]