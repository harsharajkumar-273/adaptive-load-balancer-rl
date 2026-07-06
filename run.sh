#!/bin/bash
# run.sh - Startup and installation script for the RL load balancer prototype.

# Text formatting helper
INFO='\033[0;36m'
SUCCESS='\033[0;32m'
ERROR='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${INFO}[System] Setting up the AI Load Balancer Python environment...${NC}"

# Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo -e "${ERROR}[Error] Python 3 is required but was not found. Please install Python 3.10+ and try again.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${INFO}[System] Creating virtual environment (venv)...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${ERROR}[Error] Failed to create virtual environment.${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${INFO}[System] Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${INFO}[System] Installing/updating dependencies from requirements.txt...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${ERROR}[Error] Failed to install dependencies.${NC}"
    exit 1
fi

echo -e "${SUCCESS}[System] Setup completed successfully!${NC}"
echo -e "${INFO}[System] Launching the load balancer orchestrator (FastAPI + RL Agent + Traffic Gen + Dashboard)...${NC}"
echo -e "${INFO}[System] Press Ctrl+C to terminate the application.${NC}"
sleep 1.5

# Run the system
python3 src/main.py
