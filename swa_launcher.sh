#!/bin/bash

# SWA Global Launcher
# Makes SWA CLI available system-wide on Debian

# Set project root
SWA_ROOT="/services/model-with-adpter-logic-semantic"

# Change to project directory
cd "$SWA_ROOT" || {
    echo "Error: Cannot change to SWA directory: $SWA_ROOT"
    exit 1
}

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run the SWA CLI with all arguments
python main.py "$@"