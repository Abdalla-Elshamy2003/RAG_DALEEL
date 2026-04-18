# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: pgvector is installed in PostgreSQL container, not here
# This app container only needs the Python client: psycopg

# Copy the application code
COPY ingest_app/ ./ingest_app/

# Copy the Summarization pipeline
COPY Summarization_pipeline/ ./Summarization_pipeline/

# Create data directory (will be mounted as volume)
RUN mkdir -p /app/data

# Default environment variables (can be overridden)
ENV DATA_FOLDER=/app/data
ENV DB_CONN=host=postgres port=5432 dbname=docs_ingestion user=esraa password=28102003
ENV PYTHONUNBUFFERED=1

# Default command to run the ingestion pipeline
CMD ["python", "-m", "ingest_app.main_pipeline"]