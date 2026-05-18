from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import RAGEngine
import os

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine once at startup
rag = RAGEngine()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
def root():
    return {"message": "RAG Chatbot is running!"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = rag.answer(request.question)
    return AnswerResponse(answer=result["answer"], sources=result["sources"])


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    count = rag.add_document(file_path, file.filename)
    return {"message": f"Uploaded and indexed {count} chunks from {file.filename}"}