
from flask import Flask, request, jsonify
from search import DocumentSearchEngine

app = Flask(__name__)

# Initialize the search engine once when the API starts.
search_engine = DocumentSearchEngine()

# Index the PDF when the application starts.
search_engine.index_document("data/sample.pdf")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Document Search Engine API is ready"
    })


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json(silent=True) or {}
    query = data.get("question", "").strip()

    if not query:
        return jsonify({
            "error": "Please provide a non-empty question."
        }), 400

    try:
        # Retrieve candidate passages.
        candidates = search_engine.hybrid_search(
            query,
            top_k=10
        )

        # Rerank the candidates.
        results = search_engine.rerank(
            query,
            candidates,
            top_k=5
        )

        # Extract context for Gemini.
        context_chunks = [
            result["document"] for result in results
        ]

        # Generate the grounded answer.
        answer = search_engine.rag_generator.generate_answer(
            query,
            context_chunks
        )

        return jsonify({
            "question": query,
            "answer": answer,
            "sources": [
                {
                    "passage": i,
                    "text": result["document"],
                    "rerank_score": result["rerank_score"]
                }
                for i, result in enumerate(results, start=1)
            ]
        })

    except Exception:
        app.logger.exception("Question processing failed")
        return jsonify({
            "error": "Unable to process the question. Check the server logs."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

