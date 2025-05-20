from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str

class Recommendation(BaseModel):
    name: str
    city: str
    state: str
    description: str
    rating: float
