from pypdf import PdfReader


def load_text_file(file_path):
    """
    Read a text file and return its content.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


def load_pdf_file(file_path):
    """
    Extract text from all pages of a PDF file.
    """

    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def load_document(file_path):
    """
    Load a document based on its file extension.
    """

    if file_path.lower().endswith(".txt"):
        return load_text_file(file_path)

    elif file_path.lower().endswith(".pdf"):
        return load_pdf_file(file_path)

    else:
        raise ValueError(
            "Unsupported file format. "
            "Only .txt and .pdf files are supported."
        )
