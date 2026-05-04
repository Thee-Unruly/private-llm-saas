from fastapi import Cookie, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import base64
import hashlib
import hmac
import json
import os
import re
import requests
import secrets
import time
import uuid
from mem0 import Memory

app = FastAPI(title="Memory-Enabled Ollama API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@postgres/litellm")
APP_AUTH_SECRET = os.getenv("APP_AUTH_SECRET", "")
ACCESS_TOKEN_TTL_SECONDS = int(os.getenv("ACCESS_TOKEN_TTL_SECONDS", "86400"))
AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "memory_chat_session")
TEST_UI_PATH = os.path.join(os.path.dirname(__file__), "test_ui.html")
CHAT_UI_PATH = os.path.join(os.path.dirname(__file__), "chat_ui.html")
LOGIN_UI_PATH = os.path.join(os.path.dirname(__file__), "login.html")
SIGNUP_UI_PATH = os.path.join(os.path.dirname(__file__), "signup.html")

if not APP_AUTH_SECRET:
    raise RuntimeError("APP_AUTH_SECRET must be set for authenticated multi-user access")

parsed_database_url = urlparse(POSTGRES_URL)

# Mem0 config — uses Ollama for LLM + embeddings, Postgres for vector storage
mem0_config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "gemma2:9b",
            "ollama_base_url": OLLAMA_URL
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": OLLAMA_URL
        }
    },
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "dbname": parsed_database_url.path.lstrip("/") or "litellm",
            "user": parsed_database_url.username or "user",
            "password": parsed_database_url.password or "pass",
            "host": parsed_database_url.hostname or "postgres",
            "port": parsed_database_url.port or 5432,
            "embedding_model_dims": 768
        }
    }
}
memory = Memory.from_config(mem0_config)


@app.get("/")
async def root(session_token: Optional[str] = Cookie(default=None, alias=AUTH_COOKIE_NAME)):
    destination = "/chat-ui" if _get_optional_identity_from_token(session_token) else "/login"
    return RedirectResponse(url=destination, status_code=303)


def _encode_token(payload: Dict[str, Any]) -> str:
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_segment = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=")
    signature = hmac.new(APP_AUTH_SECRET.encode("utf-8"), payload_segment, hashlib.sha256).digest()
    signature_segment = base64.urlsafe_b64encode(signature).rstrip(b"=")
    return f"{payload_segment.decode('utf-8')}.{signature_segment.decode('utf-8')}"


def _decode_token(token: str) -> Dict[str, Any]:
    try:
        payload_segment, signature_segment = token.split(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid access token") from exc

    expected_signature = hmac.new(
        APP_AUTH_SECRET.encode("utf-8"),
        payload_segment.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    actual_signature = base64.urlsafe_b64decode(signature_segment + "=" * (-len(signature_segment) % 4))
    if not hmac.compare_digest(expected_signature, actual_signature):
        raise HTTPException(status_code=401, detail="Invalid access token")

    payload_bytes = base64.urlsafe_b64decode(payload_segment + "=" * (-len(payload_segment) % 4))
    payload = json.loads(payload_bytes.decode("utf-8"))
    expires_at = payload.get("exp")
    if not isinstance(expires_at, int) or expires_at < int(time.time()):
        raise HTTPException(status_code=401, detail="Access token expired")
    return payload


def _issue_access_token(user_id: str, team_id: str, email: str) -> str:
    now = int(time.time())
    return _encode_token(
        {
            "sub": user_id,
            "team_id": team_id,
            "email": email,
            "iat": now,
            "exp": now + ACCESS_TOKEN_TTL_SECONDS,
            "jti": secrets.token_hex(8),
        }
    )


def _identity_from_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    user_id = payload.get("sub")
    team_id = payload.get("team_id")
    email = payload.get("email")
    if not isinstance(user_id, str) or not isinstance(team_id, str) or not isinstance(email, str):
        raise HTTPException(status_code=401, detail="Invalid access token")
    return {"user_id": user_id, "team_id": team_id, "email": email}


def _identity_from_token(token: str) -> Dict[str, str]:
    payload = _decode_token(token)
    return _identity_from_payload(payload)


def _get_optional_identity_from_token(token: Optional[str]) -> Optional[Dict[str, str]]:
    if not token:
        return None
    try:
        return _identity_from_token(token)
    except HTTPException:
        return None


def get_current_identity(
    authorization: Optional[str] = Header(default=None),
    session_token: Optional[str] = Cookie(default=None, alias=AUTH_COOKIE_NAME),
) -> Dict[str, str]:
    if authorization and authorization.startswith("Bearer "):
        return _identity_from_token(authorization.removeprefix("Bearer ").strip())

    cookie_identity = _get_optional_identity_from_token(session_token)
    if cookie_identity:
        return cookie_identity

    raise HTTPException(status_code=401, detail="Missing authentication")

class RegistrationRequest(BaseModel):
    email: str
    company_name: str
    max_budget: Optional[float] = 10.0


class RegistrationResponse(BaseModel):
    status: str
    data: Dict[str, str]

@app.post("/register", response_model=RegistrationResponse)
async def register(request: RegistrationRequest):
    try:
        team_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        access_token = _issue_access_token(user_id=user_id, team_id=team_id, email=request.email)
        return {
            "status": "success",
            "data": {
                "team_id": team_id,
                "user_id": user_id,
                "api_key": "",
                "access_token": access_token
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gemma2:9b"
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    memories_used: int
    user_id: str
    team_id: str
    conversation_id: Optional[str] = None


class MemoryRecord(BaseModel):
    id: str
    memory: Optional[str] = None
    hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    user_id: Optional[str] = None


class IdentityResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    team_id: str
    email: str


def _normalize_ollama_model(model_name: Optional[str]) -> str:
    if not model_name:
        return "gemma2:9b"
    if model_name == "gemma2-9b":
        return "gemma2:9b"
    return model_name


def _extract_profile_facts(text: str) -> List[str]:
    facts: List[str] = []
    normalized_text = text.strip()
    if not normalized_text:
        return facts

    name_match = re.search(r"\bmy name is\s+([A-Z][a-zA-Z'-]*)", normalized_text, flags=re.IGNORECASE)
    if name_match:
        name = name_match.group(1)
        facts.append(f"The user's name is {name}.")

    canonical_name_match = re.search(r"\bthe user's name is\s+([A-Z][a-zA-Z'-]*)", normalized_text, flags=re.IGNORECASE)
    if canonical_name_match:
        name = canonical_name_match.group(1)
        facts.append(f"The user's name is {name}.")

    inferred_name_match = re.search(
        r"\b([A-Z][a-zA-Z'-]*)\s+prefers\s+(?:concise|brief)\s+answers\b",
        normalized_text,
    )
    if inferred_name_match:
        name = inferred_name_match.group(1)
        facts.append(f"The user's name is {name}.")

    preference_patterns = [
        r"\bi prefer concise answers\b",
        r"\bkeep (?:your|the) answers concise\b",
        r"\bi prefer brief answers\b",
        r"\bkeep (?:your|the) answers brief\b",
        r"\bthe user prefers concise answers\b",
        r"\bthe user prefers brief answers\b",
        r"\b[A-Z][a-zA-Z'-]*\s+prefers\s+concise\s+answers\b",
        r"\b[A-Z][a-zA-Z'-]*\s+prefers\s+brief\s+answers\b",
    ]
    if any(re.search(pattern, normalized_text, flags=re.IGNORECASE) for pattern in preference_patterns):
        facts.append("The user prefers concise answers.")

    return facts


def _deduplicate_strings(items: List[str]) -> List[str]:
    seen = set()
    unique_items: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_items.append(item.strip())
    return unique_items


def _build_profile_summary(memory_items: List[str]) -> str:
    extracted_facts: List[str] = []
    for item in memory_items:
        stripped_item = item.strip()
        if stripped_item.startswith("The user's name is ") or stripped_item.startswith("The user prefers "):
            extracted_facts.append(stripped_item)
            continue
        extracted_facts.extend(_extract_profile_facts(item))

    profile_facts = _deduplicate_strings(extracted_facts)
    if not profile_facts:
        return ""
    return "\n".join(profile_facts)


def _is_self_knowledge_question(text: str) -> bool:
    normalized_text = text.strip().lower()
    prompts = [
        "what do you know about me",
        "who am i",
        "summarize what you know about me",
        "tell me what you know about me",
    ]
    return any(prompt in normalized_text for prompt in prompts)


def _build_profile_reply(profile_summary: str) -> str:
    profile_lines = [line.strip() for line in profile_summary.splitlines() if line.strip()]
    rendered_facts: List[str] = []
    for line in profile_lines:
        if line.startswith("The user's name is "):
            rendered_facts.append(line.replace("The user's name is ", "Your name is ").rstrip("."))
        elif line.startswith("The user prefers "):
            rendered_facts.append(line.replace("The user prefers ", "You prefer ").rstrip("."))
    if not rendered_facts:
        return "I do not know anything about you yet. Share a few details and I will remember them for later chats."
    return "\n".join(rendered_facts) + "."

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, identity: Dict[str, str] = Depends(get_current_identity)):
    try:
        filters = {"user_id": identity["user_id"], "agent_id": identity["team_id"]}
        if request.conversation_id:
            filters["run_id"] = request.conversation_id

        past_memories = memory.search(
            query=request.message,
            filters=filters,
            top_k=5
        )
        memory_text = "\n".join([m["memory"] for m in past_memories.get("results", [])])
        all_memories = memory.get_all(filters=filters)
        all_memory_text = [item["memory"] for item in all_memories.get("results", []) if item.get("memory")]
        profile_summary = _build_profile_summary(all_memory_text)

        system_prompt = (
            "You are a helpful assistant. Keep tenant data isolated and only use memories"
            " belonging to the authenticated user and team. "
            "When relevant memory is provided below, treat it as trusted context from prior"
            " conversations with this same authenticated user. "
            "If the user asks what you know about them, answer directly from that memory. "
            "If no relevant memory is provided, say naturally that you do not know anything"
            " about them yet and invite them to share details. "
            "Do not ask the user to provide memory or refer to hidden system context. "
            "Do not claim you have no memory when relevant context is provided. "
            "Keep answers concise."
        )
        if memory_text:
            system_prompt += (
                "\n\nTrusted memory for this authenticated user:\n"
                f"{memory_text}\n\nUse these memories when they help answer the question."
            )
        if profile_summary:
            system_prompt += (
                "\n\nStable user profile facts:\n"
                f"{profile_summary}\n\nPrefer these facts when the user asks about themselves."
            )

        if _is_self_knowledge_question(request.message):
            assistant_reply = _build_profile_reply(profile_summary)
        else:
            # 2. Send directly to Ollama for inference
            ollama_model = _normalize_ollama_model(request.model)
            payload = {
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.message}
                ],
                "stream": False,
            }
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            assistant_reply = response.json()["message"]["content"]

        # 3. Save exchange to memory — user_id as direct param ✅
        memory.add(
            [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": assistant_reply}
            ],
            user_id=identity["user_id"],
            agent_id=identity["team_id"],
            run_id=request.conversation_id
        )

        extracted_profile_facts = _deduplicate_strings(_extract_profile_facts(request.message))
        if extracted_profile_facts:
            memory.add(
                [{"role": "user", "content": " ".join(extracted_profile_facts)}],
                user_id=identity["user_id"],
                agent_id=identity["team_id"],
                run_id=request.conversation_id
            )

        return {
            "reply": assistant_reply,
            "memories_used": len(past_memories.get("results", [])),
            "user_id": identity["user_id"],
            "team_id": identity["team_id"],
            "conversation_id": request.conversation_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/me", response_model=IdentityResponse)
async def me(identity: Dict[str, str] = Depends(get_current_identity)):
    return identity


@app.get("/memories", response_model=Dict[str, List[MemoryRecord]])
async def get_memories(
    conversation_id: Optional[str] = None,
    identity: Dict[str, str] = Depends(get_current_identity),
):
    filters = {"user_id": identity["user_id"], "agent_id": identity["team_id"]}
    if conversation_id:
        filters["run_id"] = conversation_id
    results = memory.get_all(filters=filters)
    return results

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/test-ui")
async def test_ui():
    if not os.path.exists(TEST_UI_PATH):
        raise HTTPException(status_code=404, detail="Test UI not found")
    return FileResponse(TEST_UI_PATH)


@app.get("/chat-ui")
async def chat_ui(session_token: Optional[str] = Cookie(default=None, alias=AUTH_COOKIE_NAME)):
    if not _get_optional_identity_from_token(session_token):
        return RedirectResponse(url="/login", status_code=303)
    if not os.path.exists(CHAT_UI_PATH):
        raise HTTPException(status_code=404, detail="Chat UI not found")
    return FileResponse(CHAT_UI_PATH)


@app.get("/login")
async def login_ui(session_token: Optional[str] = Cookie(default=None, alias=AUTH_COOKIE_NAME)):
    if _get_optional_identity_from_token(session_token):
        return RedirectResponse(url="/chat-ui", status_code=303)
    if not os.path.exists(LOGIN_UI_PATH):
        raise HTTPException(status_code=404, detail="Login UI not found")
    return FileResponse(LOGIN_UI_PATH)


@app.get("/signup")
async def signup_ui(session_token: Optional[str] = Cookie(default=None, alias=AUTH_COOKIE_NAME)):
    if _get_optional_identity_from_token(session_token):
        return RedirectResponse(url="/chat-ui", status_code=303)
    if not os.path.exists(SIGNUP_UI_PATH):
        raise HTTPException(status_code=404, detail="Sign up UI not found")
    return FileResponse(SIGNUP_UI_PATH)