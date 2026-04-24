FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.3
ENV POETRY_VIRTUALENVS_CREATE=false

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi

COPY src/ ./src/
COPY tests/ ./tests/
COPY notebooks/ ./notebooks/

ENV PYTHONPATH=/app/src
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
