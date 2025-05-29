from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Optional: Fix Hugging Face Hub symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# Load the sentence transformer model only once at startup
model = SentenceTransformer('all-MiniLM-L6-v2')

# Normalize skill (lowercase, trim)
def normalize(skill):
    return skill.lower().strip()

# Calculate partial match percentage
def partial_match_percentage(query_skills, emp_skills):
    matched_skills = [q for q in query_skills if any(q in e for e in emp_skills)]
    return len(matched_skills) / len(query_skills) if query_skills else 0

@app.route("/match", methods=["POST"])
def match_employees():
    data = request.get_json()
    employees = data["employees"]
    query_skills = [normalize(s) for s in data["query_skills"]]

    # Normalize and combine employee skills
    for emp in employees:
        emp["skills"] = [normalize(s) for s in emp["skills"]]
        emp["skills_text"] = ", ".join(emp["skills"])

    # Generate embeddings for employees and query
    employee_texts = [emp["skills_text"] for emp in employees]
    employee_embeddings = model.encode(employee_texts, convert_to_tensor=True)
    query_text = ", ".join(query_skills)
    query_embedding = model.encode(query_text, convert_to_tensor=True)

    # Compute cosine similarities
    cos_sim = util.cos_sim(query_embedding, employee_embeddings)[0].cpu().numpy()

    # Weights for combining
    WEIGHT_EMBEDDING = 0.7
    WEIGHT_PARTIAL = 0.3

    combined_scores = []
    for i, emp in enumerate(employees):
        partial_score = partial_match_percentage(query_skills, emp["skills"])
        embedding_score = float(cos_sim[i])  # Make JSON serializable
        combined_score = WEIGHT_EMBEDDING * embedding_score + WEIGHT_PARTIAL * partial_score
        combined_scores.append({
            "id": emp["id"],
            "skills": emp["skills"],
            "embedding_score": embedding_score,
            "partial_score": partial_score,
            "combined_score": combined_score
        })

    # Sort employees by combined score (highest first)
    combined_scores.sort(key=lambda x: x["combined_score"], reverse=True)

    return jsonify(combined_scores)

# Run locally (or use gunicorn when deploying)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
