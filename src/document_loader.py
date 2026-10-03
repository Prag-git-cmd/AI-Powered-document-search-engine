from text_processor import clean_text, create_chunks


def load_text_file(file_path):
    """
    Read a text file and return its content as a string.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


if __name__ == "__main__":

    file_path = "data/sample.txt"

    document_text = load_text_file(file_path)

    print("Original document:")
    print(document_text)

    cleaned_text = clean_text(document_text)

    print("\nCleaned document:")
    print(cleaned_text)

    chunks = create_chunks(
        cleaned_text,
        chunk_size=30,
        overlap=5
    )

    print("\nDocument chunks:")
    print("--------------------------------")

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(chunk)
