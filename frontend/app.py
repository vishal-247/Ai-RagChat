import os
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from .backend.extract_text_and_chunk import process_pdf


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}
uploaded_docs = []  # In-memory store — replace with a vector DB in production

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        doc = {
            'id': len(uploaded_docs) + 1,
            'name': filename,
            'size': os.path.getsize(filepath),
            'path': filepath,
            'chunks': process_pdf(file),  # TODO: populate after chunking
        }
        uploaded_docs.append(doc)

        # TODO: Add your RAG pipeline here:
        
        #   1. Extract text (e.g. PyMuPDF / pdfplumber)
        #   2. Chunk the text
        #   3. Embed chunks (e.g. OpenAI / sentence-transformers)
        #   4. Store in vector DB (e.g. Chroma, Pinecone, Weaviate)

        return jsonify({'success': True, 'document': doc})

    return jsonify({'error': 'Invalid file type. Only PDFs allowed.'}), 400


@app.route('/documents', methods=['GET'])
def list_documents():
    return jsonify({'documents': uploaded_docs})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    user_message = data['message']
    history = data.get('history', [])

    # TODO: Add your RAG pipeline here:
    # 1. Retrieve relevant chunks from ChromaDB
    results = collection.query(query_texts=[user_message], n_results=4)
    chunks = results["documents"][0]        # list of matching text chunks
    metadatas = results["metadatas"][0]     # their metadata

    #   1. Embed the user query
       # 2. Build context string from retrieved chunks
    if chunks:
        context_parts = []
        for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
            source = meta.get("source", "unknown")
            context_parts.append(f"[{i}] (from {source})\n{chunk}")
        context = "\n\n".join(context_parts)
    else:
        context = "No relevant documents found."

    # 3. Build the system prompt
    system_prompt = """You are a helpful assistant that answers questions based on provided document context.
    
Rules:
- Answer using ONLY the information in the context below.
- If the context doesn't contain enough information, say so clearly.
- Always cite which source(s) you used (e.g. "According to report.pdf...").
- Be concise and accurate.

Context:
""" + context

    # 4. Build message history for Claude (convert frontend history format)
    messages = []
    for turn in history[:-1]:  # exclude the latest user message, we add it fresh
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})


    # llm call
    import os
    from google import genai
    from dotenv import load_dotenv


    load_dotenv()

    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise SystemExit(
            "Missing GEMINI_API_KEY. Create a .env file or set the environment variable."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-2.5-flash", messages=[{"role":"system","content":system_prompt}]+messages,
    )
    

    # --- Placeholder response ---

    reply = response.choices[0].message.content

    return jsonify({
        'reply': reply,
        'sources': [],  # TODO: return retrieved chunk sources
    })


@app.route('/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    global uploaded_docs
    doc = next((d for d in uploaded_docs if d['id'] == doc_id), None)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404

    try:
        os.remove(doc['path'])
    except FileNotFoundError:
        pass

    uploaded_docs = [d for d in uploaded_docs if d['id'] != doc_id]
    return jsonify({'success': True})


if __name__ == '__main__':
    app.run(debug=True, port=5000)