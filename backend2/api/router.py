# backend2/api/router.py
from utils.config import Config
from core.vectorstore import VectorStore
from core.embedding import ElementEmbedder
from core.navigator import UINavigator, MistralHFConnector
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components and config

# Initialize components (only once)
vector_store = VectorStore(dimension=Config.VECTOR_DIMENSION)
embedder = ElementEmbedder(model_name=Config.EMBEDDING_MODEL)
llm = MistralHFConnector(
    api_key=Config.HF_API_KEY,
    model=Config.LLM_MODEL,
    provider=Config.LLM_PROVIDER
)
# In api/router.py:
# Initialize navigator with RL model
navigator = UINavigator(
    vector_store=vector_store,
    embedder=embedder,
    llm=llm,
    rl_model_path="model/rl_model.pt"  # Add this
)

router = APIRouter()

# Define request/response models


class QueryRequest(BaseModel):
    query: str
    screen_metadata: Dict[str, Any]
    screenshot: Optional[str] = None


class FeedbackRequest(BaseModel):
    element_id: str
    screen_id: str
    success: bool


# Process query endpoint
@router.post("/process-query-visual")
async def process_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not request.screen_metadata.get("elements"):
        raise HTTPException(
            status_code=400, detail="No UI elements provided in screen metadata")

    result = navigator.process_query(request.query, request.screen_metadata)

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result.get(
            "error", "Failed to process query"))

    return result


# Feedback endpoint
@router.post("/action-feedback")
async def action_feedback(request: FeedbackRequest):
    success = navigator.record_feedback(
        request.element_id,
        request.screen_id,
        request.success
    )

    if not success:
        raise HTTPException(
            status_code=422, detail="Failed to record feedback")

    return {"success": True}


# Metrics endpoint
@router.get("/metrics")
async def get_metrics():
    return navigator.get_metrics()
