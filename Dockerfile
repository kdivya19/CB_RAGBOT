# # 1. Base Image
# FROM python:3.11-slim

# # 2. Work Directory
# WORKDIR /app

# # 3. Install Dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # 4. Copy Code
# # NOTE: Because of .dockerignore, 'venv' and 'helper_code.py' will be SKIPPED automatically here.
# COPY . .

# # 5. Expose & Run
# EXPOSE 8000
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


# 1. Base Image
FROM python:3.11-slim

# 2. Work Directory
WORKDIR /app

# 3. Install system deps (needed for some builds)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements
COPY requirements.txt .

# 5. Install CPU-only torch FIRST (important 🔥)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 6. Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy app code
COPY . .

# 8. Expose port
EXPOSE 8000

# 9. Run app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]