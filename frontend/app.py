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
    # print(request.files["file"])
    file = request.files['file']
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

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
            'chunks': process_pdf(filepath,filename),  # TODO: populate after chunking
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

    # 1. ChromaDB se relevant chunks lo
    import chromadb
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("documents")

    results = collection.query(query_texts=[user_message], n_results=4)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    # 2. Context banao
    if chunks:
        context_parts = []
        for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
            source = meta.get("source", "unknown")
            context_parts.append(f"[{i}] (from {source})\n{chunk}")
        context = "\n\n".join(context_parts)
    else:
        context = "No relevant documents found."

    # 3. System prompt
    system_prompt = f"""You are a helpful assistant that answers questions based on provided document context.

Rules:
- Answer using ONLY the information in the context below.
- If the context doesn't contain enough information, say so clearly.
- Always cite which source(s) you used.
- Be concise and accurate.

Context:
{context}"""

    # 4. ✅ Gemini format mein history convert karo
    gemini_messages = []
    recent_history=history[-5:-1]
    for turn in recent_history:  # last message chhodo, neeche add karenge
        role = turn.get("role", "")
        # frontend "content" key bhejta hai, Gemini ko "parts" chahiye
        text = turn.get("content") or turn.get("parts", [{}])[0].get("text", "")
        if not text:
            continue
        if role == "user":
            gemini_messages.append({"role": "user", "parts": [{"text": text}]})
        elif role in ("assistant", "model"):
            gemini_messages.append({"role": "model", "parts": [{"text": text}]})

    # Current user message add karo
    gemini_messages.append({"role": "user", "parts": [{"text": user_message}]})

    # # 5. Gemini call
    # from google import genai
    # from google.genai import types
    # from dotenv import load_dotenv
    # load_dotenv()

    # api_key = os.getenv("GEMINI_API_KEY")
    # client = genai.Client(api_key=api_key)

    # response = client.models.generate_content(
    #     model="gemini-2.0-flash-lite",
    #     contents=gemini_messages,
    #     config=types.GenerateContentConfig(
    #         system_instruction=system_prompt
    #     )
    # )

    # reply = response.text


    # 5. Groq call
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Groq OpenAI-style messages use karta hai
    groq_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": turn["role"] if turn["role"] != "model" else "assistant",
         "content": turn["parts"][0]["text"]}
        for turn in gemini_messages
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # ya "mixtral-8x7b-32768"
        messages=groq_messages,
        max_tokens=1024,
    )

    reply = response.choices[0].message.content

    sources = list(set(m.get("source", "") for m in metadatas if m.get("source")))

    return jsonify({'reply': reply, 'sources': sources})

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