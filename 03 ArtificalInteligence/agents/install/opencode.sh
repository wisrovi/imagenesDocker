#!/bin/bash

# Script: opencode.sh
# Description: Installs the opencode CLI tool from the official installer for AI-assisted coding.
# Usage: ./opencode.sh
# Prerequisites: curl, bash
# Author: AI Tools Installation Scripts
# Date: 2025-10-08
# Documentation: https://opencode.ai
# Example: After installation, use 'opencode' commands for code generation and assistance.
# Notes: Downloads and executes the installer script from opencode.ai; ensure internet connection.

# Download and execute the opencode installation script
curl -fsSL https://opencode.ai/install | bash