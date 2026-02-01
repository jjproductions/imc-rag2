        import os, re, hashlib, datetime
        from typing import List, Dict, Any, Tuple, Optional
        from PyPDF2 import PdfReader

        MD_PATTERN = re.compile(r"([#*_>`~\-]|!?\[.*?\]\(.*?\))")

        def read_text_from_file(path: str) -> Tuple[str, Optional[int]]:
            ext = os.path.splitext(path)[1].lower()
            if ext in [".txt", ".md", ".markdown"]:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if ext in [".md", ".markdown"]:
                    # naive markdown strip
                    content = MD_PATTERN.sub("", content)
                return content, None
            elif ext in [".pdf"]:
                reader = PdfReader(path)
                pages = []
                for i, p in enumerate(reader.pages):
                    try:
                        pages.append(p.extract_text() or "")
                    except Exception:
                        pages.append("")
                return "

".join(pages), None
            else:
                raise ValueError(f"Unsupported file type: {ext}")

        def recursive_find_files(root: str) -> List[str]:
            files = []
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in [".txt", ".md", ".markdown", ".pdf"]:
                        files.append(os.path.join(dirpath, fn))
            return files

        def simple_tokenize(text: str) -> List[str]:
            # whitespace tokenization approximates tokens
            return re.findall(r"\S+", text)

        def detokenize(tokens: List[str]) -> str:
            return " ".join(tokens)

        def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
            tokens = simple_tokenize(text)
            chunks = []
            i = 0
            n = len(tokens)
            if chunk_size <= 0:
                chunk_size = 800
            if overlap < 0:
                overlap = 0
            while i < n:
                chunk_tokens = tokens[i : min(i + chunk_size, n)]
                chunks.append(detokenize(chunk_tokens))
                if i + chunk_size >= n:
                    break
                i += max(1, chunk_size - overlap)
            return chunks

        def hash_text(text: str, source_path: str, chunk_id: int) -> str:
            h = hashlib.sha256()
            h.update(source_path.encode("utf-8"))
            h.update(str(chunk_id).encode("utf-8"))
            h.update(text.encode("utf-8"))
            return h.hexdigest()

        def build_payloads(source_path: str, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            payloads = []
            doc_id = os.path.relpath(source_path).replace("\", "/")
            now_iso = datetime.datetime.utcnow().isoformat() + "Z"
            for idx, ch in enumerate(chunks):
                payloads.append({
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "text": ch,
                    "source_path": source_path,
                    "page": None,
                    "hash": hash_text(ch, source_path, idx),
                    "created_at": now_iso
                })
            return payloads