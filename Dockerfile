FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/
COPY .agent/ ./.agent/
COPY scripts/ ./scripts/

# Create directories that will be mounted as volumes
RUN mkdir -p logs .data/uploads .data/results

# Install main project dependencies
RUN uv sync --frozen

# Initialize all skill virtual environments
RUN bash scripts/docker-init-skills.sh

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["uv", "run", "python", "src/webapp/app.py"]
