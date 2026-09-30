FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app:/app/src PYTHONUNBUFFERED=1

# CPU-only torch keeps the image small; drop the index line for a CUDA build.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Train on build so the image ships with models + reports ready to serve.
RUN python -m bcg.train

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
