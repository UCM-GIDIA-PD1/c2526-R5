FROM python:3.13
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml README.md ./
RUN uv sync --no-cache --no-dev
COPY . .
EXPOSE 8000
CMD ["/app/.venv/bin/fastapi", "run", "app.py", "--port", "8000"]
