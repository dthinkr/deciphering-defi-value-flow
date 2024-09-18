#!/usr/bin/env bash

# Use Python 3.11 if available, otherwise fall back to 'python3'
PYTHON=$(command -v python3.11 || command -v python3)

echo "Current Python version:"
$PYTHON --version
echo "Python path:"
which $PYTHON

echo "Please choose an option:"
echo "1. Run ingest.py"
echo "2. Run streamlit_dashboard.py"
echo "3. Run streamlit_uniswap_balance.py"
echo "4. Run endpoint.py"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Running ingest.py..."
        exec $PYTHON -m utils.ingest "$@"
        ;;
    2)
        echo "Running streamlit_dashboard.py..."
        exec streamlit run ./utils/streamlit_dashboard.py "$@"
        ;;
    3)
        echo "Running streamlit_uniswap.py..."
        exec streamlit run ./utils/streamlit_uniswap_balance.py "$@"
        ;;
    4)
        echo "Running endpoint.py..."
        exec $PYTHON -m uvicorn utils.endpoint:app --reload"$@"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac