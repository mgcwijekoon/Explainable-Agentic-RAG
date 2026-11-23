from pathlib import Path
import json
import re
import unicodedata

from src.pdf_loader import extract_text_from_pdf


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "raw" / "sample.pdf"

OUTPUT_PATH = (
    BASE_DIR / "data" / "processed" / "chunks.json"
)


class AdvancedTextChunker:

    def __init__(
        self,
        chunk_size=100,
        overlap=20,
        min_chunk_words=20
    ):

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive!")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("Invalid overlap!")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_words = min_chunk_words

    # =====================================
    # 1. TEXT CLEANING
    # =====================================

    def clean_text(self, text):

        text = unicodedata.normalize("NFKC", str(text))

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = text.replace("\x00", "")

        text = re.sub(
            r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)",
            "",
            text
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    # =====================================
    # 2. PARAGRAPH EXTRACTION
    # =====================================

    def extract_paragraphs(self, text):

        text = self.clean_text(text)

        raw_paragraphs = re.split(
            r"\n[ \t]*\n+",
            text
        )

        paragraphs = []

        for paragraph in raw_paragraphs:

            paragraph = re.sub(
                r"\s*\n\s*",
                " ",
                paragraph
            ).strip()

            if paragraph:
                paragraphs.append(paragraph)

        return paragraphs

    # =====================================
    # 3. SENTENCE EXTRACTION
    # =====================================

    def extract_sentences(self, paragraph):

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            paragraph
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # =====================================
    # 4. LONG SENTENCE SPLITTING
    # =====================================

    def split_long_sentence(self, sentence):

        words = sentence.split()

        if len(words) <= self.chunk_size:
            return [sentence]

        step = self.chunk_size - self.overlap

        pieces = []

        for start in range(0, len(words), step):

            piece = words[
                start:start + self.chunk_size
            ]

            if piece:
                pieces.append(" ".join(piece))

            if start + self.chunk_size >= len(words):
                break

        return pieces

    # =====================================
    # 5. WORD COUNT
    # =====================================

    @staticmethod
    def word_count(items):

        return sum(
            len(item["text"].split())
            for item in items
        )

    # =====================================
    # 6. OVERLAP MANAGEMENT
    # =====================================

    def create_overlap(self, previous):

        overlap_units = []

        count = 0

        for item in reversed(previous):

            overlap_units.insert(
                0,
                item
            )

            count += len(
                item["text"].split()
            )

            if count >= self.overlap:
                break

        return overlap_units

    # =====================================
    # 7. CREATE PAGE UNITS
    # =====================================

    def create_page_units(self, paragraphs):

        units = []

        for paragraph_index, paragraph in enumerate(
            paragraphs,
            start=1
        ):

            sentences = self.extract_sentences(
                paragraph
            )

            for sentence in sentences:

                pieces = self.split_long_sentence(
                    sentence
                )

                for piece in pieces:

                    units.append({

                        "text": piece,

                        "paragraph": paragraph_index

                    })

        return units

    # =====================================
    # 8. SMART PAGE CHUNKING
    # =====================================

    def build_page_chunks(self, paragraphs):

        units = self.create_page_units(
            paragraphs
        )

        chunks = []

        current = []

        for unit in units:

            unit_size = len(
                unit["text"].split()
            )

            current_size = self.word_count(
                current
            )

            # Start a new chunk when necessary
            if (
                current
                and current_size + unit_size
                > self.chunk_size
            ):

                previous = current.copy()

                chunks.append(previous)

                # Bring useful context from previous chunk
                current = self.create_overlap(
                    previous
                )

                # Maintain maximum chunk size
                while (
                    current
                    and self.word_count(current) + unit_size
                    > self.chunk_size
                ):

                    current.pop(0)

            current.append(unit)

        if current:
            chunks.append(current)

        return chunks

    # =====================================
    # 9. CREATE ALL PDF CHUNKS
    # =====================================

    def create_chunks(self, pages, source):

        all_chunks = []

        for page in pages:

            page_number = page["page"]

            if not page["text"].strip():
                continue

            paragraphs = self.extract_paragraphs(
                page["text"]
            )

            page_chunks = self.build_page_chunks(
                paragraphs
            )

            # Skip isolated cover/title-only pages.
            if (
                len(page_chunks) == 1
                and self.word_count(page_chunks[0])
                < self.min_chunk_words
            ):

                print(
                    f"Skipping short isolated page: {page_number}"
                )

                continue

            for index, chunk_items in enumerate(
                page_chunks,
                start=1
            ):

                chunk_text = " ".join(
                    item["text"]
                    for item in chunk_items
                )

                paragraph_numbers = [
                    item["paragraph"]
                    for item in chunk_items
                ]

                word_count = len(
                    chunk_text.split()
                )

                all_chunks.append({

                    "chunk_id": (
                        f"{source}_page_{page_number}_chunk_{index}"
                    ),

                    "text": chunk_text,

                    "metadata": {

                        "source": source,

                        "page": page_number,

                        "chunk_index": index,

                        "paragraph_start": min(
                            paragraph_numbers
                        ),

                        "paragraph_end": max(
                            paragraph_numbers
                        ),

                        "word_count": word_count

                    }

                })

        return all_chunks


# =====================================
# 10. MAIN FUNCTION
# =====================================

def main():

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print("Starting Improved PDF Chunking...")

    pages = extract_text_from_pdf(
        PDF_PATH
    )

    chunker = AdvancedTextChunker(
        chunk_size=100,
        overlap=20,
        min_chunk_words=20
    )

    chunks = chunker.create_chunks(
        pages,
        source=PDF_PATH.name
    )

    if not chunks:

        raise ValueError(
            "No chunks generated!"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    total_words = sum(
        chunk["metadata"]["word_count"]
        for chunk in chunks
    )

    average_words = (
        total_words / len(chunks)
    )

    print("\n" + "=" * 50)

    print("CHUNKING COMPLETED SUCCESSFULLY!")

    print("Total PDF Pages:", len(pages))

    print("Total Generated Chunks:", len(chunks))

    print(
        "Average Words:",
        round(average_words, 2)
    )

    print("Chunk Size:", chunker.chunk_size)

    print("Overlap:", chunker.overlap)

    print("=" * 50)

    print("\nFIRST FIVE CHUNKS")

    for chunk in chunks[:5]:

        print("\nChunk ID:", chunk["chunk_id"])

        print("Page:", chunk["metadata"]["page"])

        print(
            "Word Count:",
            chunk["metadata"]["word_count"]
        )

        print("Text:", chunk["text"][:400])

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":

    main()