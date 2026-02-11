"""
CV/Resume parsers for multiple file formats.
Extracts raw text from PDF, DOCX, and TXT files.
"""

import io


def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def parse_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    return "\n".join(text_parts)


def parse_txt(file_bytes: bytes) -> str:
    """Extract text from a plain text file."""
    return file_bytes.decode("utf-8", errors="ignore")


def parse_linkedin_url(url: str) -> str:
    """Extract text from a LinkedIn profile URL using trafilatura."""
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded)
        return text or ""
    return ""


def parse_file(filename: str, file_bytes: bytes) -> str:
    """
    Route file to the correct parser based on extension.
    Returns extracted raw text.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        return parse_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return parse_docx(file_bytes)
    elif ext in ("txt", "text", "csv"):
        return parse_txt(file_bytes)
    else:
        # Try as plain text fallback
        return parse_txt(file_bytes)
