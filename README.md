# Bible Study Basis

The **Bible Study Basis** is an AI-powered assistant designed to facilitate various Bible study tasks. It specializes in answering questions related to the Bible and Christianity, as well as assisting in the design and creation of Bible study materials.

Currently, the primary language supported is **Mandarin Chinese**, with plans to expand support to other languages in the future.

## Developer Notes

### 1. Environment Setup
Before running the application locally, ensure you have the necessary API keys.

1. Create a `.env` file in the root directory (refer to `.env.example` or the code for required keys like `OPENAI_API_KEY`).
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Local Development (Without Docker)
You can run the services directly on your machine for rapid development. Ensure you load the environment variables first.

*   **Load Environment Variables:**
    ```bash
    source setup_local.sh
    ```

*   **Run the MCP Server:**
    ```bash
    ./start-mcp.sh
    ```

*   **Run the UI:**
    ```bash
    ./start-ui.sh
    ```

*   **Test the Agent (CLI):**
    ```bash
    python py/agent.py
    ```

### 3. Local Development (With Docker Compose)
To test the full containerized setup locally (simulating the production environment):

```bash
docker-compose up -d --build
```

### 4. Build and Push Docker Images

To build the docker images by using `build-docker-mcp.sh` and `build-docker-ui.sh`.

To push the images to Google Artifact Registry, use `push-docker-2-gcloud.sh mcp` or `push-docker-2-gcloud.sh ui`.
