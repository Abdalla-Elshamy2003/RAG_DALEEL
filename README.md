# RAG_DALEEL

Document processing and text extraction pipeline for Retrieval-Augmented Generation (RAG) applications.

## Project Description

This project processes text files and extracts text from PDF, DOCX, and TXT documents, then stores the data in a PostgreSQL database. It is used to prepare training data for AI models in the field of search and querying.

## File Structure

```
.
├── README.md              # This file - project explanation and installation
├── TODO.md                # Remaining tasks list
├── requirements.txt       # Python requirements
├── Dockerfile             # Docker container file
├── docker-compose.yml     # Docker Compose file for running the complete project
├── .dockerignore          # Files ignored in Docker
├── data/                  # Folder containing files to be processed
└── ingest_app/            # Main application code
    ├── __init__.py
    ├── config.py          # Application and database settings
    ├── db.py              # Database functions
    ├── file_utils.py      # File processing utilities
    ├── main_pipeline.py   # Main processing script
    ├── payload_builders.py # Data structure builders for files
    ├── text_utils.py      # Text processing utilities
    └── __pycache__/       # Compiled Python files (ignored)
```

## Requirements

- Python 3.11+
- PostgreSQL
- Docker and Docker Compose (optional)

## Installation and Running

### Method 1: Using Docker Compose (Recommended)

1. **Install Docker and Docker Compose:**
   - Download and install Docker Desktop from the official website: https://www.docker.com/products/docker-desktop
   - Make sure Docker is running

2. **Download the project:**
   ```bash
   git clone <repository-url>
   cd pre_data
   ```

3. **Run the project:**
   ```bash
   docker-compose up --build
   ```

   This command will:
   - Build the Docker image for the application
   - Start a PostgreSQL database
   - Process all files in the `data/` folder
   - Save results to the database

### Method 2: Local Installation

1. **Download Python:**
   - Download Python 3.11 from the official website: https://www.python.org/downloads/
   - Make sure Python is added to PATH

2. **Download PostgreSQL:**
   - Download PostgreSQL from the official website: https://www.postgresql.org/download/
   - Or use Docker to run PostgreSQL only:
     ```bash
     docker run --name postgres -e POSTGRES_DB=docs_ingestion -e POSTGRES_USER=abdallah -e POSTGRES_PASSWORD=28102003 -p 5432:5432 -d postgres:15
     ```

3. **Download the project:**
   ```bash
   git clone <repository-url>
   cd pre_data
   ```

4. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Modify settings (optional):**
   - Open `ingest_app/config.py` and modify `db_conn` if necessary

6. **Run the project:**
   ```bash
   python -m ingest_app.main_pipeline
   ```

## Configuration

- **Data folder:** Can be changed via the `DATA_FOLDER` environment variable
- **Database settings:** Can be changed via the `DB_CONN` environment variable
- **Supported file types:** PDF, DOCX, TXT

## Troubleshooting

- Make sure Docker is running
- Make sure there are files in the `data/` folder
- Check database settings
- For Docker issues, try `docker-compose down` then `docker-compose up --build`

## Contributing

To contribute to the project:
1. Fork the project
2. Create a new branch for the feature
3. Push the changes
4. Create a Pull Request

## License

This project is licensed under the MIT License.
