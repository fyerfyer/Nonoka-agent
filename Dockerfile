FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY nonoka ./nonoka
RUN pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 nonoka
USER nonoka
EXPOSE 8000
CMD ["uvicorn", "nonoka.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
