from fastapi import APIRouter
from pydantic import BaseModel
from backend.rag_engine import SalesRAG

router = APIRouter()
rag = SalesRAG()

class QuestionRequest(BaseModel):
    question: str

@router.post("/")
def ask_question(req: QuestionRequest):
    return {"answer": rag.answer_question(req.question)}
