#!/bin/bash

# Use the PORT environment variable provided by Cloud Run, default to 8080
PORT=${PORT:-8080}

# Run the Streamlit GUI as a module from the installed package
python -m streamlit run ./src/bible_study_basis/gui.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false