FROM python:3.12-slim

# fonts for the synthetic photos of the demo robot
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[gemini]" && useradd --create-home --uid 10001 inspector
COPY config ./config

USER inspector
ENTRYPOINT ["poi-inspect", "--config", "/app/config/inspector.yaml"]
CMD ["serve"]
