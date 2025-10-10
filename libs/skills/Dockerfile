FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy skills library
COPY libs/skills/ ./libs/skills/
COPY libs/common/ ./libs/common/

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "libs.skills.registry"]
