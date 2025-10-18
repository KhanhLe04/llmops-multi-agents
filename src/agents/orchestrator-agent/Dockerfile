# ===========================
#  🚧 Stage 1: Builder
# ===========================
FROM python:3.13-slim AS builder
WORKDIR /app
COPY pyproject.toml /app/
# Install uv
RUN pip install --no-cache-dir uv==0.9.3
RUN uv pip install --no-cache-dir -r pyproject.toml --target=/app/requirements 



FROM python:3.13-slim
WORKDIR /app
# Install dependencies
COPY --from=builder /app/requirements /usr/local/lib/python3.13/site-packages/
COPY . /app
EXPOSE 7010
RUN pip install --no-cache-dir uv==0.9.3
CMD ["uv", "run", "."]