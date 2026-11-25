import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError("No content found in PDF.")

    print(f"Successfully loaded {len(documents)} pages.")

    return documents
