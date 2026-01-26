#!/bin/bash

# Determine the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ENV_FILE="${SCRIPT_DIR}/.env"

if [ -f "$ENV_FILE" ]; then
  echo "Loading secrets from ${ENV_FILE}..."
  # Read .env file, ignore comments starting with #, and export variables
  export $(grep -v '^#' "$ENV_FILE" | xargs)
else
  echo "Warning: .env file not found at ${ENV_FILE}"
fi

# Extra env variables for the local dev
# export BSB_MCP_SERVER="http://localhost:8080/mcp"
export BSB_MCP_SERVER="https://bsb-mcp-408042734994.us-east4.run.app/mcp"