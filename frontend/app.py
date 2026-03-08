import os
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

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
            'chunks': 0,  # TODO: populate after chunking
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
    #   1. Embed the user query
    #   2. Retrieve top-k relevant chunks from vector DB
    #   3. Build a context-augmented prompt
    #   4. Call your LLM (e.g. OpenAI, Claude, etc.)
    #   5. Return the response

    # --- Placeholder response ---
    reply = (
        f"[RAG Placeholder] You asked: \"{user_message}\". "
        "Connect your retrieval pipeline and LLM in app.py → /chat to get real answers from your uploaded PDFs."
    )

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