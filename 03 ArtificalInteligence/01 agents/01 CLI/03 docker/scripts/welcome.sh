#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Welcome Banner
echo -e "${BLUE}"
figlet "wAgents"
echo -e "${NC}"
echo -e "${YELLOW}Welcome to your AI Agent Development Environment!${NC}"
echo ""

# Pre-installed Tools
echo -e "${GREEN}Pre-installed Tools:${NC}"
echo -e "${BLUE}Core:${NC} python, ipython, git, curl, wget, unzip, zstd"
echo -e "${BLUE}Monitoring:${NC} btop, nvitop"
echo -e "${BLUE}AI & Dev:${NC} opencode, edit (nano), bat (cat), ruff, pre-commit"
echo ""

# Scripts
echo -e "${GREEN}Available Scripts (/scripts):${NC}"
echo -e "${BLUE}Executor Scripts (/scripts/executor):${NC}"
echo "  - auto_reload_py.sh: Auto-reload Python scripts on change."
echo "  - correct_quality_py.sh: Check and correct Python code quality."
echo "  - new_curl.sh: Enhanced curl wrapper."
echo "  - scan_code_vulnerability.sh: Scan code for vulnerabilities."
echo "  - scan_libraries_vulnerability.sh: Scan libraries for vulnerabilities."
echo "  - see_image_with_clickimage.py: View images using clickimage."
echo "  - see_imagen_with_sixel.py: View images using sixel graphics."

echo -e "${BLUE}Install Scripts (/scripts/install):${NC}"
echo "  - dvc_controller.sh: Manage DVC (Data Version Control)."
echo "  - images_control.sh: Control and manage python images."
echo "  - other_agents.sh: Manage other agent processes (gemini, copilot)."

echo ""

# Requirements
echo -e "${GREEN}Available Requirements (/requirements):${NC}"
echo "  - base.txt: Core dependencies for the environment."
echo "  - dvc.txt: Dependencies for Data Version Control."
echo "  - security.txt: Security scanning and analysis tools."
echo "  - see_image_terminal.txt: Dependencies for terminal image viewing."

echo ""
echo -e "${YELLOW}Happy Coding!${NC}"
