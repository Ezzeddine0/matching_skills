from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

app = Flask(__name__)

# Optional: Fix HF warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# Load model once at startup
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Normalization
def normalize(skill):
    return skill.lower().strip()

# Partial substring match percentage
def partial_match_percentage(query_skills, emp_skills):
    matched_skills = []
    for q_skill in query_skills:
        if any(q_skill in emp_skill for emp_skill in emp_skills):
            matched_skills.append(q_skill)
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

    # Embeddings
    employee_texts = [emp["skills_text"] for emp in employees]
    employee_embeddings = model.encode(employee_texts, convert_to_tensor=True)
    query_text = ", ".join(query_skills)
    query_embedding = model.encode(query_text, convert_to_tensor=True)

    # Similarities
    cos_sim = util.cos_sim(query_embedding, employee_embeddings)[0].cpu().numpy()

    # Weights
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

    # Sort by combined score
    combined_scores.sort(key=lambda x: x["combined_score"], reverse=True)

    return jsonify(combined_scores)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
