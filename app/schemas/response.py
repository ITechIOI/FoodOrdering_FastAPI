from pydantic import BaseModel
from typing import List
from typing import List, Dict, Any

class FoodItem(BaseModel):
    id: int
    name: str
    # description: str
    # imageUrl: str

class PredictionResponse(BaseModel):
    predicted_label: str
    similar_items: List[FoodItem]

class MatchResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]