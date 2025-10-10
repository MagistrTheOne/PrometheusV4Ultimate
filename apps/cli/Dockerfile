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

# Copy CLI application
COPY apps/cli/ ./apps/cli/

# Set Python path
ENV PYTHONPATH=/app

# Create symlink for promu command
RUN ln -s /app/apps/cli/main.py /usr/local/bin/promu
RUN chmod +x /usr/local/bin/promu

# Default command
CMD ["promu", "--help"]
