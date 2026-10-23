from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference.explain_review import explain_review


app = FastAPI(
    title="TruthLens API",
    description="AI-powered deceptive review detection API",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    review_text: str
    verified_purchase: str | None = None


@app.get("/")
def root():
    return {
        "message": "TruthLens API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "TruthLens API"
    }


@app.post("/api/analyze-review")
def analyze_review(request: ReviewRequest):

    if not request.review_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Review text cannot be empty."
        )

    try:
        result = explain_review(
            request.review_text,
            request.verified_purchase
        )

        return {
            "prediction": result["prediction"],
            "calculations": result["calculations"],
            "linguistic_features": result["linguistic_features"],
            "signals": result["signals"],
            "term_contributions": result["term_contributions"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )