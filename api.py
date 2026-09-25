import os

os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import hashlib
import time
from typing import List, Optional
import base64
from io import BytesIO
import tempfile
import shutil

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

from fastapi import FastAPI, Path, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from dotenv import load_dotenv

# NOTE: `vision`, `agronomist`, and `whisper` are deliberately NOT imported here.
# vision.py loads 5 TensorFlow models, agronomist.py initializes Earth Engine +
# Tavily + Gemini clients + builds all Agent objects, and whisper pulls in
# torch/numba — all genuinely slow/heavy. Importing them at module load time
# would block Python from even reaching `app = FastAPI(...)`, let alone
# `uvicorn.run()`. Instead: vision + agronomist load inside the `startup` event
# (after the app object exists and Uvicorn has begun booting), and whisper loads
# lazily on first transcription request. See `startup_event` below.

from rag import ingest_pdf, get_chunk_count  # lightweight at import time — no model load happens here

from db import (
    init_db,
    check_db_health,
    async_get_or_create_user,
    async_create_conversation,
    async_save_message,
    async_get_history,
    async_get_conversations_for_user,
    async_get_messages_for_conversation,
    async_session_maker,
    User,
    Conversation,
    async_delete_conversation,
    async_create_password_reset_token,
    async_update_user_password,
    async_verify_password_reset_token,
    async_use_password_reset_token,
    get_user_facts,
    save_user_fact,
    save_chat_image,
    async_delete_account,
    async_save_message_feedback
)

load_dotenv()

app = FastAPI(
    title="Agronomist AI API",
    description="Backend API for the Agronomist AI application",
    version="1.0.0",
)


# Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models

class ChatRequest(BaseModel):
    message: str
    email: str
    name: Optional[str] = None
    conversation_id: Optional[int] = None
    language: Optional[str] = None
    images: Optional[List[str]] = None  # Base64 encoded images

    class Config:
        extra = "allow"

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[dict] = None

class HistoryResponse(BaseModel):
    success: bool
    history: List[dict]
    conversations: List[dict]
    error: Optional[str] = None

class NewChatRequest(BaseModel):
    email: str
    name: Optional[str] = None

class NewChatResponse(BaseModel):
    success: bool
    conversation_id: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database: str

class ForgotPasswordRequest(BaseModel):
    email: str

class VerifyResetCodeRequest(BaseModel):
    email: str
    code: str

class ResetPasswordRequest(BaseModel):
    email: str
    code: str
    new_password: str

# class MessageFeedbackRequest(BaseModel):
#     message_id: int
#     feedback_type: str  # 'good' or 'bad'
#     comment: Optional[str] = None

class DeleteAccountRequest(BaseModel):
    email: str
    password: str  # require password confirmation for safety

class TranscribeRequest(BaseModel):
    audio: str  # Base64 encoded audio
    filename: Optional[str] = "audio.webm"

class TranscribeResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None

# Auth models
class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    user: Optional[dict] = None
    error: Optional[str] = None
    message: Optional[str] = None


FASTER_WHISPER_MODEL = None

def load_whisper_model():
    global FASTER_WHISPER_MODEL
    if FASTER_WHISPER_MODEL is None:
        from faster_whisper import WhisperModel
        print("📥 Loading Whisper model (faster-whisper, CPU)...")
        FASTER_WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")  #"base"
        print("✅ Whisper model loaded successfully!")
    return FASTER_WHISPER_MODEL


#feedback function
@app.post("/api/message/feedback")
async def submit_message_feedback(request: Request):
    try:
        body = await request.json()
        
        # Get user email from cookie
        cookieHeader = request.headers.get("cookie") or ""
        emailMatch = cookieHeader.match(r"userEmail=([^;]+)")
        email = emailMatch.group(1) if emailMatch else None
        
        if not email:
            return JSONResponse(
                status_code=401,
                content={"success": False, "error": "Not authenticated"}
            )
        
        user = await async_get_or_create_user(email, email.split("@")[0])
        
        feedback_req = MessageFeedbackRequest(**body)
        
        # Validate feedback type
        if feedback_req.feedback_type not in ['good', 'bad']:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid feedback type"}
            )
        
        # Save feedback
        success = await async_save_message_feedback(
            user.id,
            feedback_req.message_id,
            feedback_req.feedback_type,
            feedback_req.comment
        )
        
        return JSONResponse(
            status_code=200,
            content={"success": success, "message": "Feedback saved"}
        )
        
    except Exception as e:
        print(f"❌ Feedback error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
    

@app.post("/api/rag/upload")
async def upload_knowledge_pdf(file: UploadFile = File(...), email: str = Form(...)):
    """Upload a PDF to the RAG knowledge base"""
    try:
        if not file.filename.endswith(".pdf"):
            return JSONResponse(status_code=400, content={"success": False, "error": "Only PDF files allowed"})
        
        os.makedirs("./knowledge_base", exist_ok=True)
        save_path = f"./knowledge_base/{file.filename}"
        
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        chunks = ingest_pdf(save_path)
        
        return {"success": True, "message": f"Ingested {chunks} chunks from {file.filename}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/api/rag/status")
async def rag_status():
    try:
        from rag import get_chunk_count
        count = get_chunk_count()
        return {"success": True, "total_chunks": count, "ready": count > 0}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    
@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    email: str = Form(...)
):
    """Transcribe audio to text using local Whisper model (FREE)"""
    try:
        print(f"🎤 Transcription request from {email}")
        
        user = await async_get_or_create_user(email, email.split("@")[0])
        
        file_extension = ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        print(f"📁 Temporary file saved: {temp_path} ({len(content)} bytes)")
        print("FILE EXISTS:", os.path.exists(temp_path))
        print("FILE SIZE:", os.path.getsize(temp_path))

        try:
            import subprocess

            test_cmd = ["ffmpeg", "-i", temp_path, "-f", "null", "-"]
            result = subprocess.run(test_cmd, capture_output=True, text=True)
            print(result.stderr)

            model = load_whisper_model()

            print("🔄 Transcribing audio...")
            segments, info = model.transcribe(
                temp_path,
                language=None,
                beam_size=5,                     # wider search = better accuracy on short/unclear clips
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    threshold=0.3,
                    min_speech_duration_ms=100,
                    speech_pad_ms=400,
                ),
                condition_on_previous_text=False,
            )
            transcript_text = " ".join(seg.text for seg in segments).strip()
            detected_language = info.language

            # Safety net: VAD can over-trim very short/quiet clips and strip out
            # real speech entirely. If that happens, retry once without VAD
            # instead of immediately failing.
            if not transcript_text:
                print("⚠️ No speech detected with VAD filter — retrying without VAD...")
                segments, info = model.transcribe(
                    temp_path,
                    language=None,
                    beam_size=5,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                transcript_text = " ".join(seg.text for seg in segments).strip()
                detected_language = info.language

            if not transcript_text:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "No speech detected. Please speak clearly and try again."}
                )

            print(f"✅ Transcription successful (language: {detected_language}): {transcript_text[:100]}...")
            return JSONResponse(
                status_code=200,
                content={"success": True, "transcript": transcript_text, "language": detected_language}
            )
        
        finally:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"🗑️ Cleaned up temporary file")
            except Exception as e:
                print(f"⚠️ Could not delete temp file: {e}")
    
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    
######updated get conversation endpoint to return 404 if conversation not found, even if messages exist
@app.get("/api/conversation/{user_email}/{conversation_id}")
async def get_conversation(user_email: str, conversation_id: str):
    try:
        user = await async_get_or_create_user(user_email, user_email.split("@")[0])

        try:
            conv_id_int = int(conversation_id)
        except ValueError:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid conversation ID"})

        # Get conversation metadata FIRST — this is what determines 404, not message count
        async with async_session_maker() as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == conv_id_int)
            )
            conv = result.scalars().first()
            if not conv:
                return JSONResponse(status_code=404, content={"success": False, "error": "Conversation not found"})
            conv_meta = {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat() if conv.created_at else None
            }

        # Messages can legitimately be empty (e.g. a brand-new chat) — that's not an error
        messages = await async_get_messages_for_conversation(conv_id_int)
        msgs_out = [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                "conversation_id": m.conversation_id,
            }
            for m in messages
        ]

        return JSONResponse(status_code=200, content={"success": True, "conversation": conv_meta, "messages": msgs_out})
    except Exception as e:
        print(f"❌ get_conversation error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


#api for forget password
@app.post("/api/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    """Send password reset code to user's email"""
    try:
        async with async_session_maker() as session:
            result = await session.execute(select(User).where(User.email == request.email))
            user = result.scalars().first()
            
            if not user:
                return JSONResponse(
                    status_code=200,
                    content={"success": True, "message": "If the email exists, a reset code has been sent."}
                )
        
        reset_code = await async_create_password_reset_token(request.email)
        
        from email_service import email_service
        email_sent = await email_service.send_password_reset_email(request.email, reset_code)
        
        if email_sent:
            return {"success": True, "message": "Password reset code has been sent to your email."}
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Failed to send email. Please try again later."}
            )
        
    except Exception as e:
        print(f"❌ Forgot password error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": "Failed to process reset request"})

#api for verify reset code

@app.post("/api/auth/verify-reset-code")
async def verify_reset_code(request: VerifyResetCodeRequest):
    """Verify the reset code"""
    try:
        is_valid = await async_verify_password_reset_token(request.email, request.code)
        
        if is_valid:
            return {"success": True, "message": "Code verified successfully"}
        else:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid or expired reset code"})
            
    except Exception as e:
        print(f"❌ Verify reset code error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": "Failed to verify code"})

#api for reset password

@app.post("/api/auth/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """Reset user's password"""
    try:
        is_valid = await async_verify_password_reset_token(request.email, request.code)
        if not is_valid:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid or expired reset code"})
        
        if len(request.new_password) < 6:
            return JSONResponse(status_code=400, content={"success": False, "error": "Password must be at least 6 characters"})
        
        success = await async_update_user_password(request.email, request.new_password)
        if not success:
            return JSONResponse(status_code=404, content={"success": False, "error": "User not found"})
        
        await async_use_password_reset_token(request.email, request.code)
        
        return {"success": True, "message": "Password reset successfully"}
        
    except Exception as e:
        print(f"❌ Reset password error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": "Failed to reset password"})


# Password helpers

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:
        return False
    return hash_password(plain_password) == hashed_password


# Deferred heavy-module loaders — called from startup_event via asyncio.to_thread
# so the import (and everything it triggers) runs in a worker thread instead of
# blocking the event loop. Once imported, Python caches the module in sys.modules,
# so later `from vision import X` calls inside route handlers are instant lookups.

def _import_vision():
    import vision  # noqa: F401 — side effect: loads crop classifier + 4 disease models

def _import_agronomist():
    import agronomist  # noqa: F401 — side effect: inits Earth Engine, Tavily, Gemini, builds Agents


# Startup — staged, ordered, logged: DB -> RAG -> vision models -> agronomist agents

@app.on_event("startup")
async def startup_event():
    startup_start = time.perf_counter()
    print("\n🚀 FastAPI app constructed, Uvicorn is booting — starting staged service loading...\n")

    # 1. Database
    try:
        await init_db()
        healthy = await check_db_health()
        print("✅ Database initialized successfully" if healthy else "⚠️ Database connection issues")
    except Exception as e:
        print(f"⚠️ DB init warning: {e}")

    # 2. RAG — table creation + embedding model preload
    if os.getenv("SKIP_RAG_STARTUP", "false").lower() == "true":
        print("⚠️ RAG startup skipped by SKIP_RAG_STARTUP")
    else:
        try:
            from rag import init_vector_db, get_embedder

            print("📥 Creating rag_chunks table if needed...")
            await asyncio.to_thread(init_vector_db)
            print("✅ RAG table ready")

            print("📥 Preloading embedding model...")
            await asyncio.to_thread(get_embedder)
            print("✅ Embedding model preloaded")
        except Exception as e:
            print(f"⚠️ RAG init warning: {e}")

    # 3. Vision models — crop classifier + cotton/wheat/corn/rice disease models (TensorFlow, CPU)
    try:
        print("📥 Loading vision models (crop classifier + 4 disease models)...")
        await asyncio.to_thread(_import_vision)
        print("✅ Vision models ready")
    except Exception as e:
        print(f"⚠️ Vision init warning (chat will still run, but image analysis will fail): {e}")

    # 4. Agronomist agents — Earth Engine, Tavily, Gemini client, Agent definitions
    try:
        print("📥 Initializing agronomist agents (Earth Engine / Tavily / Gemini)...")
        await asyncio.to_thread(_import_agronomist)
        print("✅ Agronomist agents ready")
    except Exception as e:
        print(f"⚠️ Agronomist init warning (chat will fail until this is fixed): {e}")

    elapsed = time.perf_counter() - startup_start
    print(f"\n🎉 Startup sequence complete in {elapsed:.1f}s — API ready to serve requests.\n")


# Authentication Endpoints

@app.post("/api/auth/signup", response_model=AuthResponse)
async def signup(request: SignupRequest):
    try:
        if not request.email or not request.password:
            return AuthResponse(success=False, error="Email and password required")

        if len(request.password) < 6:
            return AuthResponse(success=False, error="Password must be at least 6 characters")

        async with async_session_maker() as session:
            result = await session.execute(select(User).where(User.email == request.email))
            existing = result.scalars().first()
            if existing:
                return AuthResponse(success=False, error="Email already registered")

            new_user = User(email=request.email, name=request.email.split("@")[0])
            setattr(new_user, "hashed_password", hash_password(request.password))
            session.add(new_user)
            await session.commit()
            await session.refresh(new_user)

            return AuthResponse(
                success=True,
                user={"email": new_user.email, "name": new_user.name},
                message="Account created successfully",
            )
    except Exception as e:
        print(f"❌ Signup error: {e}")
        return AuthResponse(success=False, error="Registration failed")
    

@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    try:
        if not request.email or not request.password:
            return AuthResponse(success=False, error="Email and password required")

        async with async_session_maker() as session:
            result = await session.execute(select(User).where(User.email == request.email))
            user = result.scalars().first()

            if not user:
                return AuthResponse(success=False, error="User not found")

            if not verify_password(request.password, getattr(user, "hashed_password", "")):
                return AuthResponse(success=False, error="Invalid credentials")

            return AuthResponse(success=True, user={"email": user.email, "name": user.name}, message="Login successful")
    except Exception as e:
        print(f"❌ Login error: {e}")
        return AuthResponse(success=False, error="Authentication failed")

@app.get("/api/auth/me", response_model=AuthResponse)
async def get_me(email: str):
    try:
        async with async_session_maker() as session:
            result = await session.execute(select(User).where(User.email == email))
            user = result.scalars().first()
            if not user:
                return AuthResponse(success=False, error="User not found")
            return AuthResponse(success=True, user={"email": user.email, "name": user.name})
    except Exception as e:
        print(f"❌ /api/auth/me error: {e}")
        return AuthResponse(success=False, error=str(e))


# Health

@app.get("/", response_model=HealthResponse)
async def root():
    db_status = "healthy" if await check_db_health() else "degraded"
    return {"status": "healthy", "service": "Agronomist AI API", "version": "1.0.0", "database": db_status}

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    db_status = "healthy" if await check_db_health() else "degraded"
    return {"status": "healthy", "service": "Agronomist AI API", "version": "1.0.0", "database": db_status}


# Chat Endpoints

@app.post("/api/chat/new", response_model=NewChatResponse)
async def start_new_chat(request: NewChatRequest):
    try:
        user = await async_get_or_create_user(request.email, request.name or request.email.split("@")[0])
        conv_id = await async_create_conversation(user.id, "New Chat")
        
        return NewChatResponse(success=True, conversation_id=str(conv_id))
    except Exception as e:
        print(f"❌ New chat error: {e}")
        return NewChatResponse(success=False, conversation_id="", error=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(request: Request):
    try:
        raw_body = await request.body()
        if not raw_body:
            return ChatResponse(
                response="Error processing request",
                conversation_id="",
                success=False,
                error="Request body is empty. Send JSON with at least 'message' and 'email' fields."
            )

        try:
            body_json = await request.json()
        except Exception:
            return ChatResponse(
                response="Error processing request",
                conversation_id="",
                success=False,
                error="Invalid JSON in request body."
            )

        # Convert conversation_id to int 
        if "conversation_id" in body_json and body_json["conversation_id"] is not None:
            try:
                if isinstance(body_json["conversation_id"], str):
                    body_json["conversation_id"] = int(body_json["conversation_id"])
            except (ValueError, TypeError):
                body_json["conversation_id"] = None

        chat_req = ChatRequest(**(body_json or {}))
        language = (chat_req.language or "en").lower()
        user = await async_get_or_create_user(chat_req.email, chat_req.name or chat_req.email.split("@")[0])

        # Handle conversation ID
        if chat_req.conversation_id:
            db_conversation_id = chat_req.conversation_id
        else:
            db_conversation_id = await async_create_conversation(user.id, "New Chat")

        # Process images if present
        detection_summary = None
        vision_results = None  
        
        if chat_req.images and len(chat_req.images) > 0:
            from vision import analyze_image_auto  # already loaded at startup — instant lookup

            for image_str in chat_req.images:
                try:
                    image_bytes = base64.b64decode(image_str)
                    await save_chat_image(user.id, db_conversation_id, image_bytes)
                except Exception as e:
                    print(f"❌ Image decode failed: {e}")
            try:
                # Always use auto pipeline: crop classifier → disease model
                start_vision = time.perf_counter()
                analysis = analyze_image_auto(chat_req.images[0], require_threshold=0.60)
                end_vision = time.perf_counter()
                print(f"⏱️ Vision analysis took {end_vision - start_vision:.2f} seconds")

                if analysis["mode"] == "rejected":
                    vision_results = analysis
                    detection_summary = (
                        f"⚠️ Unable to analyze this image.\n"
                        f"{analysis['description']}\n\n"
                        f"{analysis['advice']}"
                    )

                elif analysis["mode"] == "auto_uncertain_dual":
                    vision_results = analysis
                    detection_summary = (
                        f"⚠️ The crop classifier was uncertain.\n\n"
                        f"{analysis['description']}\n\n"
                        f"{analysis['advice']}"
                    )
                
                else:
                    status = "Healthy" if "healthy" in str(analysis['label']).lower() else "Infected"
                    vision_results = {
                        "mode": analysis.get("mode", "auto"),
                        "crop": analysis['crop'],
                        "status": status,
                        "label": analysis['label'],
                        "confidence": analysis['confidence'],
                        "description": analysis.get('description', ''),
                        "advice": analysis.get('advice', ''),
                        "uncertain": analysis.get("uncertain", False)
                    }

                    if vision_results.get("uncertain", False):
                        detection_summary = (
                            f"⚠️ The disease model was not confident in its prediction.\n"
                            f"It suggested **{vision_results['label']}**, but the confidence was only {vision_results['confidence']:.2f}.\n"
                            f"This may not be accurate. Please confirm the crop type or upload a clearer image."
                        )
                    else:
                        detection_summary = (
                            f"Crop: {vision_results['crop']}\n"
                            f"Disease: {vision_results['label']}\n\n"
                            f"{vision_results['advice']}"
                        )
            except Exception as vis_e:
                print(f"❌ Vision detection error: {vis_e}")
                import traceback
                traceback.print_exc()
                detection_summary = f"[Vision error: {str(vis_e)}]"
                vision_results = None
            

        # message for agent
        full_message = chat_req.message or "Analyze this crop image."
        
        if detection_summary:
            full_message = f"[VISION ANALYSIS]\n{detection_summary}\n\n[USER MESSAGE]\n{full_message}"

        # Save user message 
        user_message_to_save = chat_req.message or "[Image uploaded]"
        if chat_req.images:
            user_message_to_save = f"[Image] {user_message_to_save}"
        await async_save_message(user.id, "user", user_message_to_save, db_conversation_id)

        # Extract and store persistent user facts
        msg_lower = (chat_req.message or "").lower()

        if "my name is" in msg_lower:
            try:
                name_part = msg_lower.split("my name is")[-1].strip()
                name = name_part.split()[0].capitalize()
                await save_user_fact(user.id, "name", name)
            except Exception as e:
                print(f"❌ Name extraction failed: {e}")

        crop_phrases = ["i grow", "i farm", "my crops are", "i cultivate"]
        for phrase in crop_phrases:
            if phrase in msg_lower:
                try:
                    crop_part = msg_lower.split(phrase)[-1].split(".")[0].strip()
                    await save_user_fact(user.id, "crop_interests", crop_part)
                    break
                except Exception as e:
                    print(f"❌ Crop extraction failed: {e}")

        if "what's my name" in msg_lower or "do you know my name" in msg_lower:
            facts = await get_user_facts(user.id)
            name = facts.get("name", None)
            response_text = f"Your name is {name}." if name else "I don't have your name saved yet."
            await async_save_message(user.id, "assistant", response_text, db_conversation_id)
            return ChatResponse(response=response_text, conversation_id=str(db_conversation_id), success=True)

        if "what crops do i grow" in msg_lower or "what are my crops" in msg_lower or "do you remember my crops" in msg_lower:
            facts = await get_user_facts(user.id)
            crops = facts.get("crop_interests", None)
            response_text = f"You grow {crops}." if crops else "I don't have your crop interests saved yet."
            await async_save_message(user.id, "assistant", response_text, db_conversation_id)
            return ChatResponse(response=response_text, conversation_id=str(db_conversation_id), success=True)

        # Process with agronomy team — pass vision_results AND images (needed for
        # multimodal grounding and the crop-confirmation re-prediction flow)
        from agronomist import process_with_agronomy_team  # already loaded at startup — instant lookup

        start_agent = time.perf_counter()
        response_text = await process_with_agronomy_team(
            message=full_message,
            email=chat_req.email,
            conversation_id=db_conversation_id,
            language=language,
            images=chat_req.images,
            vision_data=vision_results  
        )
        end_agent = time.perf_counter()
        print(f"⏱️ Agent response took {end_agent - start_agent:.2f} seconds")

        await async_save_message(user.id, "assistant", response_text, db_conversation_id)

        # Update conversation title 
        async with async_session_maker() as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == db_conversation_id)
            )
            conv = result.scalars().first()
            if conv and conv.title == "New Chat":
                if vision_results:
                    if vision_results.get("mode") == "auto_uncertain_dual":
                        new_title = f"{vision_results['crop_1']} or {vision_results['crop_2']}"
                    elif vision_results.get("mode") == "rejected":
                        new_title = "Uncertain Crop"
                    elif vision_results.get("mode") == "auto_certain":
                        new_title = f"Uncertain - {vision_results['crop']}"
                    else:
                        new_title = f"{vision_results['crop']} - {vision_results['label']}"
                else:
                    msg = chat_req.message or "Image Analysis"
                    new_title = msg[:50] + ("..." if len(msg) > 50 else "")
                conv.title = new_title
                await session.commit()

        return ChatResponse(
            response=response_text, 
            conversation_id=str(db_conversation_id), 
            success=True,
            metadata={"vision_results": vision_results} if vision_results else None
        )

    except Exception as e:
        print(f"❌ Chat error: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            response="Error processing request", 
            conversation_id="", 
            success=False, 
            error=str(e)
        )


@app.delete("/api/conversation/{user_email}/{conversation_id}")
async def delete_conversation(user_email: str, conversation_id: str = Path(...)):
    try:
        user = await async_get_or_create_user(user_email, user_email.split("@")[0])

        try:
            conv_id_int = int(conversation_id)
        except ValueError:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid conversation ID"})

        deleted = await async_delete_conversation(conv_id_int)

        # Do not fail if already deleted
        if not deleted:
            conversations = await async_get_conversations_for_user(user.id)
            return JSONResponse(
                status_code=200,
                content={"success": True, "message": "Conversation already deleted", "conversations": conversations}
            )

        conversations = await async_get_conversations_for_user(user.id)
        return JSONResponse(status_code=200, content={"success": True, "conversations": conversations})

    except Exception as e:
        print(f"❌ delete_conversation error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

# History & Conversation Endpoints 

@app.get("/api/history/{user_email}", response_model=HistoryResponse)
async def get_chat_history(user_email: str, limit: int = 20):
    try:
        user = await async_get_or_create_user(user_email, user_email.split("@")[0])

        conversations = await async_get_conversations_for_user(user.id)
        conversations.sort(
            key=lambda x: x.get("last_message_at") or x.get("created_at") or "",
            reverse=True,
        )

        db_messages = await async_get_history(user.id, limit=limit)
        history_data = [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "conversation_id": msg.conversation_id,
            }
            for msg in db_messages
        ]

        return HistoryResponse(success=True, conversations=conversations, history=history_data)

    except Exception as e:
        print(f"❌ History error: {e}")
        return HistoryResponse(success=False, conversations=[], history=[], error=str(e))


@app.get("/api/conversation/{user_email}/{conversation_id}")
async def get_conversation(user_email: str, conversation_id: str):
    try:
        user = await async_get_or_create_user(user_email, user_email.split("@")[0])

        try:
            conv_id_int = int(conversation_id)
        except ValueError:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid conversation ID"})

        messages = await async_get_messages_for_conversation(conv_id_int)
        if not messages:
            return JSONResponse(status_code=404, content={"success": False, "error": "No messages found for this conversation"})

        msgs_out = [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                "conversation_id": m.conversation_id,
            }
            for m in messages
        ]

        async with async_session_maker() as session:
            result = await session.execute(select(Conversation).where(Conversation.id == conv_id_int))
            conv = result.scalars().first()
            if conv:
                conv_meta = {
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat() if conv.created_at else None
                }
            else:
                return JSONResponse(status_code=404, content={"success": False, "error": "Conversation not found"})

        return JSONResponse(status_code=200, content={"success": True, "conversation": conv_meta, "messages": msgs_out})
    except Exception as e:
        print(f"❌ get_conversation error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.delete("/api/auth/delete-account")
async def delete_account(request: DeleteAccountRequest):
    """Permanently delete user account and archive all data"""
    try:
        async with async_session_maker() as session:
            result = await session.execute(select(User).where(User.email == request.email))
            user = result.scalars().first()

            if not user:
                return JSONResponse(status_code=404, content={"success": False, "error": "User not found"})

            if not verify_password(request.password, getattr(user, "hashed_password", "")):
                return JSONResponse(status_code=401, content={"success": False, "error": "Incorrect password"})

            user_id = user.id

        success = await async_delete_account(user_id)

        if success:
            return {"success": True, "message": "Account deleted successfully"}
        else:
            return JSONResponse(status_code=500, content={"success": False, "error": "Failed to delete account"})

    except Exception as e:
        print(f"❌ Delete account error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# Session Utilities

@app.post("/api/user")
async def create_or_get_user(request: ChatRequest):
    try:
        user = await async_get_or_create_user(request.email, request.name or request.email.split("@")[0])
        return {"success": True, "user": {"id": user.id, "email": user.email, "name": user.name}}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Main runner

def main():
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

if __name__ == "__main__":
    main()