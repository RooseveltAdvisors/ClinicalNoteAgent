#!/bin/bash
# Docker initialization script for skills
# Called during Docker build or container startup

set -e

echo "=== Docker: Initializing Clinical Notes Agent Skills ==="

# Change to project directory
cd /app

# Run skills setup
bash .agent/setup_skills.sh

echo "=== Docker: Skills initialization complete ==="
