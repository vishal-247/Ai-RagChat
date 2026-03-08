
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(filepath):
    # 1. Extract
    doc = fitz.open(filepath)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n\n"

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)

    # 3. TODO: embed each chunk and store in vector DB
    return chunks