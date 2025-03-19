from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    chat_history: list[tuple[str, str]]

class ChatResponse(BaseModel):
    response: str 
