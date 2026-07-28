FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.4.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global user.email "docker@tech-challenge.com" \
    && git config --global user.name "Docker Build"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root

COPY . .

COPY .env.example .env

RUN git init && git add -A && git commit -m "docker-init"

CMD ["poetry", "run", "dvc", "repro"]