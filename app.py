from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import uuid
import os
from mem0 import Memory

app = FastAPI(title="LiteLLM SaaS Registration API")

# Configuration from environment variables
LITELLM_URL = os.getenv("LITELLM_URL", "http://litellm:4000")
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-A2PqYnyEcNmuRSMQerWgiDIs")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@postgres/litellm")

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
            "dbname": "litellm",
            "user": "user",
            "password": "pass",
            "host": "postgres",
            "port": 5432
        }
    }
}
memory = Memory.from_config(mem0_config)

class LiteLLMSaaSManager:
    def __init__(self, base_url=LITELLM_URL, master_key=MASTER_KEY):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {master_key}",
            "Content-Type": "application/json"
        }

    def create_team(self, team_name: str, max_budget: float = 10.0, models: List[str] = ["gemma4-e4b"]):
        url = f"{self.base_url}/team/new"
        data = {
            "team_alias": team_name,
            "max_budget": max_budget,
            "models": models
        }
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()['team_id']

    def create_user(self, email: str, team_id: Optional[str] = None):
        url = f"{self.base_url}/user/new"
        user_id = str(uuid.uuid4())
        data = {
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
            "models": ["gemma4-e4b"]
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

@app.post("/register")
async def register(request: RegistrationRequest):
    try:
        # 1. Create Team
        team_id = manager.create_team(team_name=request.company_name, max_budget=request.max_budget)
        
        # 2. Create User
        user_id = manager.create_user(email=request.email, team_id=team_id)
        
        # 3. Generate Key
        api_key = manager.generate_key_for_user(
            user_id=user_id, 
            team_id=team_id, 
            key_alias=f"{request.company_name} Primary Key"
        )
        
        return {
            "status": "success",
            "data": {
                "team_id": team_id,
                "user_id": user_id,
                "api_key": api_key
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    user_id: str
    message: str
    model: Optional[str] = "gemma2-9b"

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 1. Retrieve relevant past memories for this user
        past_memories = memory.search(query=request.message, filters={"user_id": request.user_id}, limit=5)
        memory_text = "\n".join([m["memory"] for m in past_memories.get("results", [])])

        system_prompt = "You are a helpful assistant."
        if memory_text:
            system_prompt += f"\n\nRelevant context from past conversations:\n{memory_text}"

        # 2. Send to LiteLLM (which routes to Ollama)
        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ]
        }
        response = requests.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {MASTER_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=120
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        assistant_reply = response.json()["choices"][0]["message"]["content"]

        # 3. Save this exchange to memory
        memory.add(
            [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": assistant_reply}
            ],
            user_id=request.user_id
        )

        return {"reply": assistant_reply, "memories_used": len(past_memories.get("results", []))}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
