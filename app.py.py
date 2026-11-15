import os
import zipfile
import chromadb
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from your deployed frontend

# ---------------------------------------------------------
# 1. Unzip & Load Serialized Vector Database
# ---------------------------------------------------------
DB_DIR = "./aurorastay_chroma_db"
DB_ZIP = "./aurorastay_chroma_db.zip"

if not os.path.exists(DB_DIR) and os.path.exists(DB_ZIP):
    with zipfile.ZipFile(DB_ZIP, "r") as zf:
        zf.extractall(".")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_collection(name="aurorastay_policies")

# ---------------------------------------------------------
# 2. Initialize Gemini LLM Client & Grounding Prompt
# ---------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
llm_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

LLM_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.1-flash-lite"]

SYSTEM_PROMPT = """You are an internal policy assistant for AuroraStay Hotels staff.
Answer staff questions accurately and concisely using ONLY the provided policy context.

RULES:
1. Grounding: Use ONLY facts explicitly stated in the Retrieved Context. Do not guess or invent rules, fees, or deadlines.
2. Missing Info: If the context does not contain the answer, say: "I cannot find that information in the provided AuroraStay policy documents. Please check the live property profile or escalate to a manager."
3. Policy Precedence: Always include important conditions or exceptions mentioned in the policy (e.g., "the reservation confirmation controls if it states a different rule", "subject to availability", or "verify the live property profile").
4. Citations: Cite the supporting document title and page number for your statements (e.g., [Room and Property Information Guide | Page 4])."""


def retrieve_context(query: str, top_k: int = 5):
    query_embedding = embed_model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved_chunks.append({
            "text": doc,
            "metadata": meta,
            "similarity_score": round(1 - dist, 4),
        })
    return retrieved_chunks


def generate_answer(query: str, retrieved_chunks: list) -> str:
    if not llm_client:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        header = f"[Source {i}: {meta['document_title']} ({meta['policy_code']}) | Page {meta['page']}]"
        context_parts.append(f"{header}\n{chunk['text']}")

    formatted_context = "\n\n---\n\n".join(context_parts)
    prompt = (
        f"Retrieved Policy Context:\n{formatted_context}\n\n"
        f"Staff Question:\n{query}\n\n"
        "Answer the question based strictly on the context above and cite the supporting policy document(s) and page number(s)."
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
    )

    for model_name in LLM_MODELS:
        try:
            response = llm_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            return response.text
        except (ServerError, ClientError) as e:
            if any(code in str(e) for code in ["429", "503", "404", "per day"]):
                continue
            raise e

    raise RuntimeError("All configured Gemini models are temporarily unavailable.")


# ---------------------------------------------------------
# 3. Flask API Endpoints
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "AuroraStay Hotels Policy Assistant API",
        "status": "running",
        "indexed_chunks": collection.count(),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "chunks": collection.count()}), 200


@app.route("/query", methods=["POST"])
def query_policy():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    top_k = int(data.get("top_k", 5))

    if not question:
        return jsonify({"error": "Please provide a non-empty 'question' field."}), 400

    try:
        retrieved_chunks = retrieve_context(question, top_k=top_k)
        answer = generate_answer(question, retrieved_chunks)

        # Deduplicate sources for clean UI display
        sources = []
        seen = set()
        for c in retrieved_chunks:
            meta = c["metadata"]
            key = (meta["policy_code"], meta["page"])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "policy_code": meta["policy_code"],
                    "document_title": meta["document_title"],
                    "page": meta["page"],
                    "similarity_score": c["similarity_score"],
                    "excerpt": c["text"],
                })

        return jsonify({
            "question": question,
            "answer": answer,
            "sources": sources,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)