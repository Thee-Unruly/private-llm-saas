# Self-Hosted LLM Backend (LiteLLM + Ollama)

A self-hosted backend for running local large language models (LLMs) using LiteLLM and Ollama. It provides secure API endpoints for model inference, user authentication, team key management, and is fully containerized for easy deployment.

## Features
- Local LLM inference with Ollama
- LiteLLM API compatibility
- User authentication and team key management
- Dockerized deployment with `docker-compose`
- Easily extensible backend

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Thee-Unruly/private-llm-saas.git
   cd VLLM
   ```

2. **Configure your environment:**
   - Copy `litellm_config.example.yaml` to `litellm_config.yaml` and set your own `master_key` and other secrets.

3. **Start the services:**
   ```sh
   docker-compose up --build
   ```

4. **Access the API:**
   - The backend will be available at the address specified in your `docker-compose.yml`.

## Security
- Do NOT commit `litellm_config.yaml` or any private keys to version control.
- Always use strong, unique secrets for your `master_key` and authentication.

## Folder Structure
- `app.py` — Main backend application
- `saas_registration.py` — User/team management logic
- `ollama_data/` — Model blobs and SSH keys (excluded from git)
- `litellm_config.yaml` — Secrets and config (excluded from git)

## License
Specify your license here.
