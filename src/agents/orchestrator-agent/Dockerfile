# ===========================
#  🚧 Stage 1: Builder
# ===========================
FROM python:3.13-slim AS build

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1 
ENV UV_LINK_MODE=copy
# Set working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv==0.9.3

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

COPY . .
EXPOSE 7010

CMD ["uv", "run", "."]