#!/bin/bash

# Source the environment setup script
source setup_env.sh

# Use the PORT environment variable provided by Cloud Run, default to 8080
PORT=${PORT:-8080}

# Run the Streamlit GUI
# This command blocks, keeping the Docker container alive.
python -m streamlit run py/gui.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false