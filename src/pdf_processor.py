
import re
import fitz


def extract_pdf_pages(pdf_bytes: bytes):
    """
    Extract text page by page.
    Returns a list of dictionaries with page number and text.
    """
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            # Sort text blocks approximately in reading order.
            blocks = page.get_text("blocks", sort=True)
            cleaned_blocks = []

            for block in blocks:
                if len(block) < 5:
                    continue

                block_text = block[4]
                if not isinstance(block_text, str):
                    continue

                block_text = block_text.replace("\u00a0", " ")
                block_text = re.sub(r"[ \t]+", " ", block_text)
                block_text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", block_text)
                block_text = re.sub(r"\n{3,}", "\n\n", block_text)
                block_text = block_text.strip()

                if block_text:
                    cleaned_blocks.append(block_text)

            page_text = "\n\n".join(cleaned_blocks).strip()

            if page_text:
                pages.append({
                    "page": page_number,
                    "text": page_text,
                })

    return pages
