from pathlib import Path
import json
import numpy as np
import chromadb


BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"

EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "embeddings.npz"

DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "ds205_documents"


def store_embeddings():

    # =====================================
    # STEP 1: CHECK FILES
    # =====================================

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("chunks.json not found!")

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError("embeddings.npz not found!")

    # =====================================
    # STEP 2: LOAD CHUNKS
    # =====================================

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    if not chunks:
        raise ValueError("No chunks found!")

    # =====================================
    # STEP 3: LOAD EMBEDDINGS
    # =====================================

    with np.load(EMBEDDINGS_PATH) as data:

        ids = data["ids"].astype(str).tolist()

        embeddings = data["embeddings"].astype(
            np.float32
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D matrix!"
        )

    if len(chunks) != len(ids):
        raise ValueError(
            "Chunk and ID counts do not match!"
        )

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Chunk and embedding counts do not match!"
        )

    if len(set(ids)) != len(ids):
        raise ValueError(
            "Duplicate chunk IDs detected!"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain invalid values!"
        )

    # Verify ID consistency
    expected_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if ids != expected_ids:
        raise ValueError(
            "Chunk IDs and embedding IDs do not match!"
        )

    # =====================================
    # STEP 4: PREPARE DOCUMENTS
    # =====================================

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [

        {
            "source": chunk["metadata"]["source"],

            "page": chunk["metadata"]["page"],

            "chunk_index": chunk["metadata"]["chunk_index"]

        }

        for chunk in chunks

    ]

    # =====================================
    # STEP 5: CONNECT CHROMADB
    # =====================================

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # =====================================
    # STEP 6: REMOVE OLD COLLECTION
    # =====================================

    existing_collections = client.list_collections()

    existing_names = {
        item.name if hasattr(item, "name") else str(item)
        for item in existing_collections
    }

    if COLLECTION_NAME in existing_names:

        print("Removing old collection...")

        client.delete_collection(
            name=COLLECTION_NAME
        )

    # =====================================
    # STEP 7: CREATE FRESH COLLECTION
    # =====================================

    collection = client.create_collection(

        name=COLLECTION_NAME,

        metadata={
            "hnsw:space": "cosine"
        }

    )

    # =====================================
    # STEP 8: STORE NEW EMBEDDINGS
    # =====================================

    print("Storing new embeddings...")

    collection.add(

        ids=ids,

        embeddings=embeddings.tolist(),

        documents=documents,

        metadatas=metadatas

    )

    # =====================================
    # STEP 9: VERIFY STORAGE
    # =====================================

    stored_count = collection.count()

    if stored_count != len(chunks):

        raise RuntimeError(
            "Vector database verification failed!"
        )

    print("\nVECTOR DATABASE UPDATED SUCCESSFULLY!")

    print("Total Chunks:", len(chunks))

    print("Stored Documents:", stored_count)

    print("Vector Dimensions:", embeddings.shape[1])

    print("Database Location:", DB_PATH)

    print("Collection:", COLLECTION_NAME)


if __name__ == "__main__":

    store_embeddings()