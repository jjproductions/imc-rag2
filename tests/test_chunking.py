from rag_api.app.utils.chunking import chunk_text

def test_chunking_basic():
    text = " ".join(["token"] * 2000)
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    # Expect ceil((2000-800)/(800-100)) + 1 ~= 3 or 4
    assert len(chunks) >= 3
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)