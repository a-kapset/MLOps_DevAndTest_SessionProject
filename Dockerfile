FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_CACHE_DIR=/tmp/poetry-cache

RUN python -m venv /opt/poetry && /opt/poetry/bin/pip install "poetry==2.4.1"

WORKDIR /app
COPY pyproject.toml poetry.lock README.md ./
RUN /opt/poetry/bin/poetry install --only main --no-root && rm -rf $POETRY_CACHE_DIR

COPY src ./src
RUN /opt/poetry/bin/poetry install --only main && rm -rf $POETRY_CACHE_DIR

COPY models/model.joblib ./models/model.joblib

RUN useradd --create-home --uid 1000 app
USER app

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "credit_default.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
