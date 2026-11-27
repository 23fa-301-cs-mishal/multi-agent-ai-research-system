import requests
import pymupdf
def extract_pdf_text(url):
    """Download a PDF and extract all its text."""

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    pdf = pymupdf.open(
        stream=response.content,
        filetype="pdf"
    )

    text = ""

    for page in pdf:
        text += f"{page.get_text('text')}\n"

    pdf.close()

    return text