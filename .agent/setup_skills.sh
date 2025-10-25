#!/bin/bash
# Setup script to initialize uv environments for all skills
# Usage: bash .agent/setup_skills.sh

set -e

SKILLS_DIR=".agent/skills"

echo "=== Clinical Notes Agent - Skills Setup ==="
echo "Initializing uv virtual environments for all skills..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Loop through all skill directories
for skill_dir in "$SKILLS_DIR"/*/ ; do
    if [ -d "$skill_dir" ]; then
        skill_name=$(basename "$skill_dir")
        
        # Skip __pycache__ and hidden directories
        if [[ "$skill_name" == "__pycache__" || "$skill_name" == .* ]]; then
            continue
        fi
        
        echo "---"
        echo "Setting up skill: $skill_name"
        
        # Check if pyproject.toml exists
        if [ ! -f "$skill_dir/pyproject.toml" ]; then
            echo "  SKIP: No pyproject.toml found"
            continue
        fi
        
        cd "$skill_dir"
        
        # Initialize virtual environment
        if [ ! -d ".venv" ]; then
            echo "  Creating virtual environment..."
            uv venv
        else
            echo "  Virtual environment exists"
        fi
        
        # Sync dependencies
        echo "  Syncing dependencies..."
        uv sync
        
        echo "  ✓ $skill_name ready"
        
        cd - > /dev/null
    fi
done

echo ""
echo "=== Setup Complete ==="
echo "All skills initialized successfully!"
echo ""
echo "To use a skill:"
echo "  cd .agent/skills/{skill-name}"
echo "  uv run python {script_name}.py"
