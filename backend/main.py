import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import json
import faiss
import numpy as np

def dummy_embed(text: str) -> np.ndarray:
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(512).astype("float32")

with open("shl_assessment_actual.json", "r") as f:
    assessments = json.load(f)

assessment_texts = [a["description"] for a in assessments]
embeddings = np.vstack([dummy_embed(text) for text in assessment_texts])

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

class QueryInput(BaseModel):
    query: str

class AssessmentResult(BaseModel):
    name: str
    url: str
    remote_testing: str
    adaptive_irt: str
    duration_minutes: int
    test_type: str

class RecommendResponse(BaseModel):
    recommendations: List[AssessmentResult]

app = FastAPI(title="SHL Assessment Recommendation API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the SHL Assessment Recommendation API"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend_assessments(query_input: QueryInput):
    query = query_input.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    query_embedding = dummy_embed(query).reshape(1, -1)
    
    k = 10
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for idx in indices[0]:
        if idx < len(assessments):
            a = assessments[idx]
            results.append(
                AssessmentResult(
                    name=a["name"],
                    url=a["url"],
                    remote_testing=a["remote_testing"],
                    adaptive_irt=a["adaptive_irt"],
                    duration_minutes=a["duration_minutes"],
                    test_type=a["test_type"]
                )
            )
    return RecommendResponse(recommendations=results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
