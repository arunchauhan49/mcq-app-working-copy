import os
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Literal, Dict

from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, EmailStr, ConfigDict

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

from auth import (
    hash_password,
    verify_password,
    create_token,
    get_current_user_id,
)
from llm_service import extract_text_from_file, generate_mcqs

UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

app = FastAPI()
api = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============== MODELS ==============

class RegisterInput(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str = Field(min_length=1)


class LoginInput(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    user: Dict


class NoteOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    chapter: str
    filename: Optional[str] = None
    source_type: str
    preview: str
    content_length: int
    created_at: str


class MCQOptions(BaseModel):
    A: str
    B: str
    C: str
    D: str


class MCQOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    session_id: str
    note_id: Optional[str] = None
    chapter: str
    question: str
    options: MCQOptions
    correct_option: Literal["A", "B", "C", "D"]
    explanation: str
    selected_option: Optional[Literal["A", "B", "C", "D"]] = None
    is_correct: Optional[bool] = None
    answered_at: Optional[str] = None
    created_at: str


class AnswerInput(BaseModel):
    selected_option: Literal["A", "B", "C", "D"]


class GenerateInput(BaseModel):
    note_id: str
    count: int = Field(ge=0, le=60)  # 0 = auto (fully text-driven)
    allow_outside: bool = False


class MixedGenerateInput(BaseModel):
    count: int = Field(ge=0, le=60)  # 0 = auto
    note_ids: Optional[List[str]] = None
    allow_outside: bool = False


# ============== AUTH ==============

@api.post("/auth/register", response_model=AuthResponse)
async def register(payload: RegisterInput):
    existing = await db.users.find_one({"email": payload.email.lower()})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = str(uuid.uuid4())
    doc = {
        "id": user_id,
        "email": payload.email.lower(),
        "name": payload.name,
        "password_hash": hash_password(payload.password),
        "created_at": now_iso(),
    }
    await db.users.insert_one(doc)
    token = create_token(user_id, doc["email"])
    return AuthResponse(
        token=token, user={"id": user_id, "email": doc["email"], "name": doc["name"]}
    )


@api.post("/auth/login", response_model=AuthResponse)
async def login(payload: LoginInput):
    user = await db.users.find_one({"email": payload.email.lower()})
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user["id"], user["email"])
    return AuthResponse(
        token=token,
        user={"id": user["id"], "email": user["email"], "name": user["name"]},
    )


@api.get("/auth/me")
async def me(user_id: str = Depends(get_current_user_id)):
    user = await db.users.find_one({"id": user_id}, {"_id": 0, "password_hash": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ============== NOTES ==============

@api.post("/notes/upload", response_model=NoteOut)
async def upload_note(
    chapter: str = Form(...),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user_id),
):
    if not chapter.strip():
        raise HTTPException(status_code=400, detail="Chapter name required")

    note_id = str(uuid.uuid4())
    content = ""
    filename = None
    source_type = "text"

    if file and file.filename:
        filename = file.filename
        ext = Path(filename).suffix.lower()
        allowed = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"}
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(sorted(allowed))}",
            )
        user_dir = UPLOAD_DIR / user_id
        user_dir.mkdir(exist_ok=True)
        saved_path = user_dir / f"{note_id}{ext}"
        data = await file.read()
        saved_path.write_bytes(data)
        source_type = "pdf" if ext == ".pdf" else ("image" if ext in {".png", ".jpg", ".jpeg", ".webp"} else "text")
        try:
            content = await extract_text_from_file(str(saved_path), filename)
        except Exception as e:
            logger.exception("Extraction failed")
            raise HTTPException(status_code=500, detail=f"Text extraction failed: {e}")
    elif text and text.strip():
        content = text.strip()
        source_type = "text"
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or text")

    if not content.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the note")

    doc = {
        "id": note_id,
        "user_id": user_id,
        "chapter": chapter.strip(),
        "filename": filename,
        "source_type": source_type,
        "content": content,
        "created_at": now_iso(),
    }
    await db.notes.insert_one(doc)
    return NoteOut(
        id=note_id,
        chapter=doc["chapter"],
        filename=filename,
        source_type=source_type,
        preview=content[:280],
        content_length=len(content),
        created_at=doc["created_at"],
    )


@api.get("/notes", response_model=List[NoteOut])
async def list_notes(user_id: str = Depends(get_current_user_id)):
    docs = await db.notes.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).to_list(500)
    return [
        NoteOut(
            id=d["id"],
            chapter=d["chapter"],
            filename=d.get("filename"),
            source_type=d["source_type"],
            preview=(d.get("content") or "")[:280],
            content_length=len(d.get("content") or ""),
            created_at=d["created_at"],
        )
        for d in docs
    ]


@api.delete("/notes/{note_id}")
async def delete_note(note_id: str, user_id: str = Depends(get_current_user_id)):
    res = await db.notes.delete_one({"id": note_id, "user_id": user_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")
    # Preserve MCQs in history — do NOT cascade delete
    return {"ok": True}


# ============== MCQ GENERATION ==============

async def _persist_mcqs(user_id: str, session_id: str, session_type: str, mcqs: List[Dict], note_id: Optional[str]) -> List[Dict]:
    docs = []
    for m in mcqs:
        mcq_id = str(uuid.uuid4())
        doc = {
            "id": mcq_id,
            "user_id": user_id,
            "session_id": session_id,
            "session_type": session_type,
            "note_id": note_id,
            "chapter": m["chapter"],
            "question": m["question"],
            "options": m["options"],
            "correct_option": m["correct_option"],
            "explanation": m["explanation"],
            "selected_option": None,
            "is_correct": None,
            "answered_at": None,
            "created_at": now_iso(),
        }
        docs.append(doc)
    if docs:
        await db.mcqs.insert_many(docs)
    return docs


def _mcq_to_out(d: Dict) -> MCQOut:
    return MCQOut(
        id=d["id"],
        session_id=d["session_id"],
        note_id=d.get("note_id"),
        chapter=d["chapter"],
        question=d["question"],
        options=MCQOptions(**d["options"]),
        correct_option=d["correct_option"],
        explanation=d["explanation"],
        selected_option=d.get("selected_option"),
        is_correct=d.get("is_correct"),
        answered_at=d.get("answered_at"),
        created_at=d["created_at"],
    )


@api.post("/mcqs/generate", response_model=List[MCQOut])
async def generate_from_note(payload: GenerateInput, user_id: str = Depends(get_current_user_id)):
    note = await db.notes.find_one({"id": payload.note_id, "user_id": user_id}, {"_id": 0})
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    try:
        mcqs = await generate_mcqs(
            [{"chapter": note["chapter"], "content": note["content"]}],
            payload.count,
            allow_outside_knowledge=payload.allow_outside,
        )
    except Exception as e:
        logger.exception("MCQ generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    session_id = str(uuid.uuid4())
    docs = await _persist_mcqs(user_id, session_id, "chapter", mcqs, payload.note_id)
    return [_mcq_to_out(d) for d in docs]


@api.post("/mcqs/generate-mixed", response_model=List[MCQOut])
async def generate_mixed(payload: MixedGenerateInput, user_id: str = Depends(get_current_user_id)):
    query = {"user_id": user_id}
    if payload.note_ids:
        query["id"] = {"$in": payload.note_ids}
    notes = await db.notes.find(query, {"_id": 0}).to_list(500)
    if not notes:
        raise HTTPException(status_code=400, detail="No notes available. Upload notes first.")

    try:
        mcqs = await generate_mcqs(
            [{"chapter": n["chapter"], "content": n["content"]} for n in notes],
            payload.count,
            allow_outside_knowledge=payload.allow_outside,
        )
    except Exception as e:
        logger.exception("Mixed MCQ generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    session_id = str(uuid.uuid4())
    docs = await _persist_mcqs(user_id, session_id, "mixed", mcqs, None)
    return [_mcq_to_out(d) for d in docs]


@api.get("/mcqs/session/{session_id}", response_model=List[MCQOut])
async def get_session(session_id: str, user_id: str = Depends(get_current_user_id)):
    docs = await db.mcqs.find(
        {"session_id": session_id, "user_id": user_id}, {"_id": 0}
    ).sort("created_at", 1).to_list(200)
    if not docs:
        raise HTTPException(status_code=404, detail="Session not found")
    return [_mcq_to_out(d) for d in docs]


@api.post("/mcqs/{mcq_id}/answer", response_model=MCQOut)
async def answer_mcq(mcq_id: str, payload: AnswerInput, user_id: str = Depends(get_current_user_id)):
    mcq = await db.mcqs.find_one({"id": mcq_id, "user_id": user_id}, {"_id": 0})
    if not mcq:
        raise HTTPException(status_code=404, detail="MCQ not found")
    if mcq.get("selected_option"):
        return _mcq_to_out(mcq)

    is_correct = payload.selected_option == mcq["correct_option"]
    ts = now_iso()
    await db.mcqs.update_one(
        {"id": mcq_id, "user_id": user_id},
        {"$set": {
            "selected_option": payload.selected_option,
            "is_correct": is_correct,
            "answered_at": ts,
        }},
    )
    mcq["selected_option"] = payload.selected_option
    mcq["is_correct"] = is_correct
    mcq["answered_at"] = ts
    return _mcq_to_out(mcq)


# ============== HISTORY ==============

@api.get("/mcqs/history", response_model=List[MCQOut])
async def history(
    chapter: Optional[str] = None,
    session_type: Optional[str] = None,
    user_id: str = Depends(get_current_user_id),
):
    query = {"user_id": user_id}
    if chapter and chapter != "all":
        query["chapter"] = chapter
    if session_type and session_type != "all":
        query["session_type"] = session_type
    docs = await db.mcqs.find(query, {"_id": 0}).sort("created_at", -1).to_list(2000)
    return [_mcq_to_out(d) for d in docs]


@api.get("/mcqs/chapters")
async def chapters(user_id: str = Depends(get_current_user_id)):
    chs = await db.mcqs.distinct("chapter", {"user_id": user_id})
    return sorted(chs)


# ============== PROGRESS ==============

@api.get("/progress")
async def progress(user_id: str = Depends(get_current_user_id)):
    pipeline = [
        {"$match": {"user_id": user_id, "selected_option": {"$ne": None}}},
        {
            "$group": {
                "_id": "$chapter",
                "total": {"$sum": 1},
                "correct": {"$sum": {"$cond": ["$is_correct", 1, 0]}},
            }
        },
    ]
    by_chapter = []
    async for row in db.mcqs.aggregate(pipeline):
        by_chapter.append(
            {
                "chapter": row["_id"],
                "total": row["total"],
                "correct": row["correct"],
                "incorrect": row["total"] - row["correct"],
                "accuracy": round(100 * row["correct"] / row["total"], 1) if row["total"] else 0,
            }
        )

    total = sum(c["total"] for c in by_chapter)
    correct = sum(c["correct"] for c in by_chapter)
    total_generated = await db.mcqs.count_documents({"user_id": user_id})
    return {
        "total_answered": total,
        "total_correct": correct,
        "total_incorrect": total - correct,
        "overall_accuracy": round(100 * correct / total, 1) if total else 0,
        "total_generated": total_generated,
        "by_chapter": sorted(by_chapter, key=lambda x: x["chapter"].lower()),
    }


# ============== HEALTH ==============

@api.get("/")
async def root():
    return {"message": "MCQ API up"}


app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db():
    client.close()
