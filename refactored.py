from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Disable Hugging Face symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# Use a smaller model (512MB RAM or less)
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
model = SentenceTransformer(MODEL_NAME)

def normalize(skill):
    """
    Normalize skill (lowercase, trim).
    """
    return skill.lower().strip()

def partial_match_percentage(query_skills, emp_skills):
    """
    Calculate partial match percentage between query and employee skills.
    """
    matched = [q for q in query_skills if any(q in e for e in emp_skills)]
    return len(matched) / len(query_skills) if query_skills else 0

@app.route("/match", methods=["POST"])
def match_employees():
    data = request.get_json()
    employees = data.get("employees", [])
    query_skills = [normalize(s) for s in data.get("query_skills", [])]

    # Prepare employee data
    for emp in employees:
        emp["skills"] = [normalize(s) for s in emp.get("skills", [])]
        emp["skills_text"] = ", ".join(emp["skills"])

    # Create embeddings
    employee_texts = [emp["skills_text"] for emp in employees]
    employee_embeddings = model.encode(employee_texts, convert_to_tensor=True)
    query_text = ", ".join(query_skills)
    query_embedding = model.encode(query_text, convert_to_tensor=True)

    # Compute cosine similarities
    cos_sim = util.cos_sim(query_embedding, employee_embeddings)[0].cpu().numpy()

    # Weights
    WEIGHT_EMBEDDING = 0.7
    WEIGHT_PARTIAL = 0.3

    # Compute combined scores
    combined_scores = []
    for i, emp in enumerate(employees):
        partial_score = partial_match_percentage(query_skills, emp["skills"])
        embedding_score = float(cos_sim[i])
        combined_score = WEIGHT_EMBEDDING * embedding_score + WEIGHT_PARTIAL * partial_score
        combined_scores.append({
            "id": emp["id"],
            "skills": emp["skills"],
            "combined_score": combined_score * 100
        })

    # Sort by combined score
    combined_scores.sort(key=lambda x: x["combined_score"], reverse=True)

    return jsonify(combined_scores)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
