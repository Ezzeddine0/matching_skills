from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Avoid warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# Load sentence-transformers model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Normalize skills
def normalize(skill):
    return skill.lower().strip()

# Partial substring match percentage
def partial_match_percentage(query_skills, emp_skills):
    matched_skills = []
    for q_skill in query_skills:
        if any(q_skill in emp_skill for emp_skill in emp_skills):
            matched_skills.append(q_skill)
    return len(matched_skills) / len(query_skills) if query_skills else 0

@app.route('/match', methods=['POST'])
def match_employees():
    data = request.get_json()
    employees = data["employees"]
    query_skills = data["query_skills"]

    # Normalize query skills
    query_skills = [normalize(s) for s in query_skills]
    query_text = ", ".join(query_skills)

    # Normalize employee skills
    for emp in employees:
        emp["skills"] = [normalize(s) for s in emp["skills"]]
        emp["skills_text"] = ", ".join(emp["skills"])

    # Create embeddings for employees and query
    employee_texts = [emp["skills_text"] for emp in employees]
    employee_embeddings = model.encode(employee_texts, convert_to_tensor=True)
    query_embedding = model.encode(query_text, convert_to_tensor=True)

    # Compute cosine similarities (semantic similarity)
    cos_sim = util.cos_sim(query_embedding, employee_embeddings)[0].cpu().numpy()

    # Combine scores
    WEIGHT_EMBEDDING = 0.7
    WEIGHT_PARTIAL = 0.3

    combined_scores = []
    for i, emp in enumerate(employees):
        partial_score = partial_match_percentage(query_skills, emp["skills"])
        embedding_score = cos_sim[i]
        combined_score = WEIGHT_EMBEDDING * embedding_score + WEIGHT_PARTIAL * partial_score

        # Convert NumPy float32 to native float
        combined_scores.append({
            "id": emp["id"],
            "name": emp.get("name", ""),
            "skills": emp["skills"],
            "embedding_score": float(embedding_score),
            "partial_score": float(partial_score),
            "combined_score": float(combined_score)
        })

    # Sort by combined score
    combined_scores.sort(key=lambda x: x["combined_score"], reverse=True)

    return jsonify(combined_scores)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
