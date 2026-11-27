
import requests
import pymupdf
import re


def remove_references(text):
    """
    Remove the references/bibliography section
    from the extracted academic paper text.
    """

    reference_headings = [
        r"references",
        r"bibliography",
        r"works cited",
        r"literature cited",
        r"references and notes",
        r"references cited",
    ]

    pattern = re.compile(
        r"(?im)^[ \t]*(?:\d+(?:\.\d+)*[.)]?\s+)?(?:"
        + "|".join(reference_headings)
        + r")[ \t]*:?[ \t]*$"
    )

    # Only detect headings in the final 45% of the document.
    minimum_position = int(len(text) * 0.55)

    for match in pattern.finditer(text):

        if match.start() >= minimum_position:

            print("References section detected and removed.")

            return text[:match.start()].rstrip()

    print("No references heading detected.")

    return text


def extract_pdf_text(url):
    """Download a PDF and extract its text without saving it to disk."""

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    pdf = pymupdf.open(
        stream=response.content,
        filetype="pdf"
    )

    text = ""

    try:
        for page in pdf:
            text += str(page.get_text("text")) + "\n"
    finally:
        pdf.close()

    # Remove references after extracting the complete PDF.
    text = remove_references(text)

    return text
