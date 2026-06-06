import os
import requests
import jwt
import datetime
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =========================
# APP SETUP
# =========================
app = Flask(__name__, static_folder="static")
app.config["SECRET_KEY"] = "medirag-secret-key-change-in-production"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///medirag.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

CORS(app)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# =========================
# MODELS
# =========================
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(100), nullable=False)
    email    = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created  = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    messages = db.relationship("Message", backref="user", lazy=True)

class Message(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    role      = db.Column(db.String(10), nullable=False)
    content   = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

with app.app_context():
    db.create_all()

# =========================
# JWT DECORATOR
# =========================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Token missing"}), 401
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            current_user = User.query.get(data["user_id"])
            if not current_user:
                return jsonify({"error": "User not found"}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# =========================
# LOAD PDFs
# =========================
print("Loading PDFs...")
folder_path = "data"
docs = []
for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder_path, file))
        docs.extend(loader.load())
print(f"Pages loaded: {len(docs)}")

# =========================
# CHUNKING
# =========================
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Chunks created: {len(chunks)}")

# =========================
# EMBEDDINGS & VECTOR DB
# =========================
print("Building vector DB...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embeddings)
print("Vector DB ready.")

# =========================
# RAG HELPERS
# =========================
def retrieve(query):
    results = vector_db.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in results])

def generate(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "phi4-mini:latest", "prompt": prompt, "stream": False},
        timeout=120
    )
    res.raise_for_status()
    return res.json()["response"]

# =========================
# AUTH ROUTES
# =========================
@app.route("/api/signup", methods=["POST"])
def signup():
    data     = request.get_json()
    name     = data.get("name", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not name or not email or not password:
        return jsonify({"error": "All fields are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409
    hashed = bcrypt.generate_password_hash(password).decode("utf-8")
    user = User(name=name, email=email, password=hashed)
    db.session.add(user)
    db.session.commit()
    token = jwt.encode(
        {"user_id": user.id, "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)},
        app.config["SECRET_KEY"], algorithm="HS256"
    )
    return jsonify({"token": token, "name": user.name}), 201

@app.route("/api/login", methods=["POST"])
def login():
    data     = request.get_json()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")
    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Invalid email or password"}), 401
    token = jwt.encode(
        {"user_id": user.id, "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)},
        app.config["SECRET_KEY"], algorithm="HS256"
    )
    return jsonify({"token": token, "name": user.name})

# =========================
# PROFILE ROUTES
# =========================
@app.route("/api/me", methods=["GET"])
@token_required
def get_me(current_user):
    return jsonify({"name": current_user.name, "email": current_user.email})

@app.route("/api/me", methods=["PUT"])
@token_required
def update_me(current_user):
    data  = request.get_json()
    name  = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    if not name or not email:
        return jsonify({"error": "Name and email are required"}), 400
    existing = User.query.filter_by(email=email).first()
    if existing and existing.id != current_user.id:
        return jsonify({"error": "Email already in use"}), 409
    current_user.name  = name
    current_user.email = email
    db.session.commit()
    return jsonify({"name": current_user.name, "email": current_user.email})

@app.route("/api/me/password", methods=["PUT"])
@token_required
def change_password(current_user):
    data     = request.get_json()
    current  = data.get("current_password", "")
    new_pass = data.get("new_password", "")
    if not bcrypt.check_password_hash(current_user.password, current):
        return jsonify({"error": "Current password is incorrect"}), 401
    if len(new_pass) < 6:
        return jsonify({"error": "New password must be at least 6 characters"}), 400
    current_user.password = bcrypt.generate_password_hash(new_pass).decode("utf-8")
    db.session.commit()
    return jsonify({"message": "Password updated"})

# =========================
# CHAT ROUTES
# =========================
@app.route("/api/chat", methods=["POST"])
@token_required
def chat(current_user):
    data  = request.get_json()
    query = data.get("message", "").strip()
    if not query:
        return jsonify({"error": "Empty message"}), 400
    context = retrieve(query)
    prompt = f"""You are a helpful medical assistant. Answer ONLY using the provided context.
If the answer is not in the context, say "I don't have information about that in my knowledge base."

Context:
{context}

Question: {query}

Answer:"""
    try:
        answer = generate(prompt)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not connect to Ollama. Make sure it is running on localhost:11434."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    db.session.add(Message(user_id=current_user.id, role="user", content=query))
    db.session.add(Message(user_id=current_user.id, role="bot",  content=answer))
    db.session.commit()
    return jsonify({"answer": answer})

@app.route("/api/history", methods=["GET"])
@token_required
def history(current_user):
    msgs = Message.query.filter_by(user_id=current_user.id).order_by(Message.timestamp).all()
    return jsonify([
        {"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()}
        for m in msgs
    ])

@app.route("/api/history", methods=["DELETE"])
@token_required
def clear_history(current_user):
    Message.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({"message": "History cleared"})

# =========================
# SERVE FRONTEND
# =========================
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path and not path.startswith("api"):
        file_path = os.path.join(app.static_folder, path)
        if os.path.exists(file_path):
            return send_from_directory("static", path)
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
