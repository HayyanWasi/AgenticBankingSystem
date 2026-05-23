# Agentic Banking System - Backend API

This directory contains the backend implementation of the Agentic Banking System. 

For the complete multi-agent system architecture diagrams, database schemas, and tool specifications, please refer to the main [README.md](../README.md) at the root of the project workspace.

---

## 🛠️ Quick Start

### 1. Ingest Privacy PDF for RAG Agent
Ensure the file `/data/privacy_policy_for_banking_simulation_app.pdf` is present, then build/populate the Chroma vector store:
```bash
uv run python -m app.agents.privacy_policy_agent.agent
```

### 2. Run the FastAPI Server
Launch the development server:
```bash
uv run uvicorn app.main:app --reload --port 8000
```
The interactive Swagger API documentation will be available at `http://localhost:8000/docs`.

### 3. Run Agents via CLI (Testing)
You can test agent graph flows directly in your terminal:

*   **Bank Manager (Supervisor)**:
    ```bash
    uv run python -m app.agents.bank_manager.agent
    ```
*   **KYC Agent**:
    ```bash
    uv run python -m app.graphs.kyc_agent_graph
    ```
*   **Loan Agent**:
    ```bash
    uv run python -m app.graphs.loan_agent_graph
    ```
*   **Transfer Agent**:
    ```bash
    uv run python -m app.graphs.transfer_graph
    ```
