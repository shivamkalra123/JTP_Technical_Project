FROM python:3.11-slim

WORKDIR /app

# First install uv (the fast pip alternative)
RUN pip install uv

COPY requirements.txt .

RUN uv pip install --system -r requirements.txt

COPY . .
COPY .env .env

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
