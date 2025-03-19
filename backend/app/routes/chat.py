from fastapi import APIRouter, HTTPException
from app.services.chat_service import ChatService
from app.models.chat_model import ChatRequest, ChatResponse

router = APIRouter()
chat_service = ChatService()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = chat_service.process_query(request.query, request.chat_history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))