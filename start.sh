#!/usr/bin/env bash
# CrowdSafe AI - Smart Launcher
# Checks the basic runtime dependencies before starting the application.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   CrowdSafe AI — Intelligent Monitor  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python3 -c "import cv2, numpy, scipy, fastapi" 2>/dev/null; then
    echo -e "${RED}Missing dependencies detected${NC}"
    echo ""
    echo "Run one of these commands first:"
    echo ""
    echo -e "  ${GREEN}./setup.sh${NC}              # Full setup with YOLO model"
    echo -e "  ${GREEN}./install-quick.sh${NC}      # Minimal dependencies"
    echo ""
    exit 1
fi

echo -e "${GREEN}All dependencies ready!${NC}"
echo ""

MODE="LIVE CAMERA"
for arg in "$@"; do
    if [ "$arg" = "--sim" ]; then
        MODE="SIMULATION"
        break
    fi
done

echo -e "${GREEN}Dashboard:${NC} http://localhost:${API_PORT:-8000}"
echo -e "${YELLOW}Mode:${NC} $MODE"
echo -e "${YELLOW}Controls:${NC} Press Q to quit, H to hide window"
echo ""
echo "Starting..."
echo ""

python3 main.py "$@"
