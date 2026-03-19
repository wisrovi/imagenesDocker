#!/bin/bash

# Chaos Testing Script for Docker-in-Docker project
# Uses Chaos Toolkit to test system resilience

set -e

echo "🌪️ Running chaos tests..."

CHAOS_VERSION="1.16.0"

# Install Chaos Toolkit if not available
if ! command -v chaos &> /dev/null; then
    echo "📦 Installing Chaos Toolkit..."
    pip install chaostoolkit chaostoolkit-docker
fi

# Run chaos experiment
echo "🧪 Running chaos experiment..."
chaos run chaos/experiment.json --journal-path chaos/journal.json

# Analyze results
echo "📊 Analyzing chaos test results..."
if grep -q '"status": "failed"' chaos/journal.json; then
    echo "❌ Chaos test revealed weaknesses in the system"
    echo "Check chaos/journal.json for details"
    exit 1
else
    echo "✅ System passed chaos testing!"
fi

echo "📈 Chaos test report: chaos/journal.json"