import requests
import json
import uuid

class LiteLLMSaaSManager:
    def __init__(self, base_url="http://localhost:7072", master_key="sk-A2PqYnyEcNmuRSMQerWgiDIs"):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {master_key}",
            "Content-Type": "application/json"
        }

    def create_team(self, team_name, max_budget=10.0, models=["gemma4-e4b"]):
        """Creates a new 'Organization' or Team for a customer."""
        url = f"{self.base_url}/team/new"
        data = {
            "team_alias": team_name,
            "max_budget": max_budget,
            "models": models
        }
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        result = response.json()
        print(f"Team '{team_name}' created. ID: {result['team_id']}")
        return result['team_id']

    def create_user(self, email, team_id=None):
        """Registers a new user and optionally assigns them to a team."""
        url = f"{self.base_url}/user/new"
        user_id = str(uuid.uuid4())
        data = {
            "user_email": email,
            "user_id": user_id
        }
        if team_id:
            data["teams"] = [team_id]
        
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        print(f"User '{email}' registered. User ID: {user_id}")
        return user_id

    def generate_key_for_user(self, user_id, team_id=None, key_alias="Default Key"):
        """Generates a new API key linked to a specific user and team."""
        url = f"{self.base_url}/key/generate"
        data = {
            "user_id": user_id,
            "key_alias": key_alias,
            "models": ["gemma4-e4b"] # Can be dynamic
        }
        if team_id:
            data["team_id"] = team_id

        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        result = response.json()
        print(f"API Key generated for user {user_id}: {result['key']}")
        return result['key']

def register_new_customer(email, company_name):
    """Workflow: Create Team -> Create User -> Generate Initial Key"""
    manager = LiteLLMSaaSManager()
    
    print(f"--- Registering {company_name} ({email}) ---")
    
    # 1. Create a Team for the company
    team_id = manager.create_team(team_name=company_name)
    
    # 2. Register the user under that team
    user_id = manager.create_user(email=email, team_id=team_id)
    
    # 3. Give them their first key
    api_key = manager.generate_key_for_user(user_id=user_id, team_id=team_id, key_alias=f"{company_name} Primary Key")
    
    return {
        "team_id": team_id,
        "user_id": user_id,
        "api_key": api_key
    }

if __name__ == "__main__":
    # Example usage:
    try:
        new_customer = register_new_customer(
            email="ceo@awesome-startup.com", 
            company_name="Awesome-Startup-Inc"
        )
        print("\n--- Registration Successful ---")
        print(json.dumps(new_customer, indent=4))
    except Exception as e:
        print(f"Error during registration: {e}")
