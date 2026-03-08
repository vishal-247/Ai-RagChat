import fitz
import chromadb
import uuid

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

def process_pdf(filepath, filename):
    doc = fitz.open(filepath)
    full_text = "".join(page.get_text() + "\n\n" for page in doc)

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(full_text)

    collection.add(
        documents=chunks,
        metadatas=[{"source": filename} for _ in chunks],
        ids=[str(uuid.uuid7()) for _ in chunks]
    )
    return len(chunks)