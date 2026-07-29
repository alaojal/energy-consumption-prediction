# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
COPY requirements.txt .

# 4. Install uv (dependency manager)
RUN pip Install uv
RUN uv sync --frozen --no-dev

# 5. Copy the entire project into the image
COPY ..

# 6. Expose FASTAPI to default port
Expose 8000

# 7. Command to run API with Uvicorn
CMD ["uv", "run", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]