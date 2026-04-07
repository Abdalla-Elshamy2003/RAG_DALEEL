# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ingest_app/ ./ingest_app/

# Create data directory (will be mounted as volume)
RUN mkdir -p /app/data

# Default environment variables (can be overridden)
ENV DATA_FOLDER=/app/data
ENV DB_CONN=host=localhost port=5432 dbname=docs_ingestion user=abdallah password=28102003

# Default command to run the ingestion pipeline
CMD ["python", "-m", "ingest_app.main_pipeline"]