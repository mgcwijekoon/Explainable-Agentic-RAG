from pathlib import Path
import re
import unicodedata

from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "raw" / "sample.pdf"


class PDFExtractionError(Exception):
    """Raised when PDF extraction cannot be completed."""
    pass


def clean_extracted_text(text):

    if not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null and soft-hyphen characters
    text = text.replace("\x00", "")
    text = text.replace("\u00ad", "")

    # Fix words broken across lines
    text = re.sub(
        r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])",
        "",
        text
    )

    # Normalize spaces
    lines = []

    for line in text.split("\n"):

        line = re.sub(
            r"[ \t]+",
            " ",
            line
        ).strip()

        lines.append(line)

    # Preserve paragraph boundaries
    cleaned_lines = []

    previous_blank = False

    for line in lines:

        if line:

            cleaned_lines.append(line)

            previous_blank = False

        else:

            if cleaned_lines and not previous_blank:

                cleaned_lines.append("")

                previous_blank = True

    return "\n".join(cleaned_lines).strip()


def extract_text_from_pdf(pdf_path, password=None):

    pdf_path = Path(pdf_path)

    # =====================================
    # STEP 1: FILE VALIDATION
    # =====================================

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if not pdf_path.is_file():

        raise ValueError(
            "Provided path is not a file!"
        )

    # =====================================
    # STEP 2: OPEN PDF
    # =====================================

    try:

        reader = PdfReader(
            str(pdf_path),
            strict=False
        )

    except Exception as error:

        raise PDFExtractionError(
            f"Unable to read PDF: {error}"
        ) from error

    # =====================================
    # STEP 3: CHECK ENCRYPTION
    # =====================================

    if reader.is_encrypted:

        try:

            decrypt_result = reader.decrypt(
                password or ""
            )

            if not decrypt_result:

                raise PDFExtractionError(
                    "PDF is encrypted. A valid password is required."
                )

        except PDFExtractionError:

            raise

        except Exception as error:

            raise PDFExtractionError(
                f"PDF decryption failed: {error}"
            ) from error

    # =====================================
    # STEP 4: PAGE EXTRACTION
    # =====================================

    extracted_pages = []

    total_pages = len(reader.pages)

    if total_pages == 0:

        raise PDFExtractionError(
            "PDF contains no pages!"
        )

    extraction_errors = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            raw_text = page.extract_text() or ""

            cleaned_text = clean_extracted_text(
                raw_text
            )

        except Exception as error:

            extraction_errors.append(
                (page_number, str(error))
            )

            print(
                f"Warning: Page {page_number} extraction failed: {error}"
            )

            cleaned_text = ""

        # Preserve original page numbers
        extracted_pages.append({

            "page": page_number,

            "text": cleaned_text

        })

    # =====================================
    # STEP 5: EXTRACTION SUMMARY
    # =====================================

    empty_pages = [
        page["page"]
        for page in extracted_pages
        if not page["text"]
    ]

    extracted_characters = sum(
        len(page["text"])
        for page in extracted_pages
    )

    print("\nPDF EXTRACTION SUMMARY")

    print("=" * 45)

    print("PDF:", pdf_path.name)

    print("Total Pages:", total_pages)

    print(
        "Successfully Processed Pages:",
        total_pages - len(extraction_errors)
    )

    print(
        "Pages With Text:",
        total_pages - len(empty_pages)
    )

    print(
        "Empty Pages:",
        len(empty_pages)
    )

    print(
        "Extracted Characters:",
        extracted_characters
    )

    if empty_pages:

        print(
            "Warning: Pages without selectable text:",
            empty_pages
        )

    if extraction_errors:

        print(
            "Pages With Extraction Errors:",
            len(extraction_errors)
        )

    print("=" * 45)

    return extracted_pages


# =====================================
# MAIN FUNCTION
# =====================================

def main():

    print("Starting Advanced PDF Processing...")

    pages = extract_text_from_pdf(
        PDF_PATH
    )

    print("\nPDF PROCESSING SUCCESSFUL!")

    print("Total Pages:", len(pages))

    print("\nEXTRACTED PAGE PREVIEW")

    for page in pages:

        print(
            f"\nPage {page['page']}"
        )

        if page["text"]:

            print(
                page["text"][:300]
            )

        else:

            print(
                "No selectable text found."
            )

    print("\nPDF Processing Completed!")


if __name__ == "__main__":

    main()