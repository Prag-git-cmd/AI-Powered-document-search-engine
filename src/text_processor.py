import re


def clean_text(text):
    """
    Clean extracted document text.

    Handles:
    - Words broken across PDF line breaks
    - Newline characters
    - Multiple spaces
    - Empty lines
    """

    # Join words that were split by a hyphen at a line break.
    # Example:
    # "prom-\nising" -> "promising"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Replace remaining newlines with spaces.
    text = text.replace("\n", " ")

    # Remove multiple spaces.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_chunks(text, chunk_size=100, overlap=20):
    """
    Split text into overlapping word-based chunks.

    chunk_size:
        Maximum number of words in a chunk.

    overlap:
        Number of words shared between consecutive chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start = end - overlap

    return chunks
