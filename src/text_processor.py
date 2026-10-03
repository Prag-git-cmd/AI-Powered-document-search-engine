def clean_text(text):
    """
    Clean unnecessary spaces and blank lines from the document.
    """

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if line:
            cleaned_lines.append(line)

    return " ".join(cleaned_lines)


def create_chunks(text, chunk_size=100, overlap=20):
    """
    Split text into overlapping chunks.

    chunk_size: maximum number of words in one chunk.
    overlap: number of words shared between consecutive chunks.
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
