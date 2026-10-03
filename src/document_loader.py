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

    print("Document loaded successfully!")
    print("--------------------------------")
    print(document_text)
