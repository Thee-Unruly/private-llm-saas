from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import base64
import hashlib
import hmac
import json
import os
import requests
import secrets
import time
import uuid
from mem0 import Memory

app = FastAPI(title="LiteLLM SaaS Registration API")

# Configuration from environment variables
LITELLM_URL = os.getenv("LITELLM_URL", "http://litellm:4000")
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-change-me-master-key")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@postgres/litellm")
APP_AUTH_SECRET = os.getenv("APP_AUTH_SECRET", "")
ACCESS_TOKEN_TTL_SECONDS = int(os.getenv("ACCESS_TOKEN_TTL_SECONDS", "86400"))

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


def get_current_identity(authorization: Optional[str] = Header(default=None)) -> Dict[str, str]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    payload = _decode_token(authorization.removeprefix("Bearer ").strip())
    user_id = payload.get("sub")
    team_id = payload.get("team_id")
    email = payload.get("email")
    if not isinstance(user_id, str) or not isinstance(team_id, str) or not isinstance(email, str):
        raise HTTPException(status_code=401, detail="Invalid access token")
    return {"user_id": user_id, "team_id": team_id, "email": email}

class LiteLLMSaaSManager:
    def __init__(self, base_url=LITELLM_URL, master_key=MASTER_KEY):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {master_key}",
            "Content-Type": "application/json"
        }

    def create_team(self, team_name: str, max_budget: float = 10.0, models: Optional[List[str]] = None):
        url = f"{self.base_url}/team/new"
        model_list = models or ["gemma2-9b"]
        data = {
            "team_alias": team_name,
            "max_budget": max_budget,
            "models": model_list
        }
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()['team_id']

    def create_user(self, email: str, team_id: Optional[str] = None):
        url = f"{self.base_url}/user/new"
        user_id = str(uuid.uuid4())
        data: Dict[str, Any] = {
            "user_email": email,
            "user_id": user_id
        }
        if team_id:
            data["teams"] = [team_id]

        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return user_id

    def generate_key_for_user(self, user_id: str, team_id: Optional[str] = None, key_alias: str = "Default Key"):
        url = f"{self.base_url}/key/generate"
        data = {
            "user_id": user_id,
            "key_alias": key_alias,
            "models": ["gemma2-9b"]
        }
        if team_id:
            data["team_id"] = team_id

        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()['key']

manager = LiteLLMSaaSManager()

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

        system_prompt = (
            "You are a helpful assistant. Keep tenant data isolated and only use memories"
            " belonging to the authenticated user and team. "
            "When relevant memory is provided below, treat it as trusted context from prior"
            " conversations with this same authenticated user. "
            "If the user asks what you know about them, answer directly from that memory. "
            "Do not claim you have no memory when relevant context is provided. "
            "Keep answers concise."
        )
        if memory_text:
            system_prompt += (
                "\n\nTrusted memory for this authenticated user:\n"
                f"{memory_text}\n\nUse these memories when they help answer the question."
            )

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