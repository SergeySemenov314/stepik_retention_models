FROM python:3.11-slim

WORKDIR /app

COPY model_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_service/app.py .
COPY model_service/models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
