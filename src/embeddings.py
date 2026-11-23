from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"

OUTPUT_PATH = BASE_DIR / "data" / "processed" / "embeddings.npz"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def generate_embeddings():

    # Check chunks file
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("chunks.json not found!")

    # Load chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No text chunks found!")

    texts = [chunk["text"] for chunk in chunks]

    # Load embedding model
    print("Loading Sentence Transformer model...")

    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Save embeddings
    np.savez_compressed(
        OUTPUT_PATH,
        ids=np.array([chunk["chunk_id"] for chunk in chunks]),
        embeddings=embeddings.astype(np.float32)
    )

    print("\nEMBEDDINGS GENERATED SUCCESSFULLY!")

    print("Total Chunks:", len(chunks))
    print("Embedding Shape:", embeddings.shape)
    print("Vector Dimensions:", embeddings.shape[1])
    print("Saved File:", OUTPUT_PATH)


if __name__ == "__main__":
    generate_embeddings()