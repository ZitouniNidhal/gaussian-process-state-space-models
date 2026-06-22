# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the pyproject.toml and source code first to leverage Docker cache
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install .[test,docs] notebook

# Copy the rest of the application code
COPY . .

# Expose port for Jupyter Notebook
EXPOSE 8888

# Default command is to run tests
CMD ["pytest"]
