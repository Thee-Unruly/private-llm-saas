from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import uuid
import os

app = FastAPI(title="LiteLLM SaaS Registration API")

# Configuration from environment variables
LITELLM_URL = os.getenv("LITELLM_URL", "http://litellm:4000")
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-A2PqYnyEcNmuRSMQerWgiDIs")

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

@app.get("/health")
async def health():
    return {"status": "healthy"}
