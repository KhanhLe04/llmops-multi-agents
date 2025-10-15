# ===========================
#  🚧 Stage 1: Builder
# ===========================
FROM python:3.13-slim AS builder

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv==0.9.3


# Copy pyproject and lock file
COPY pyproject.toml uv.lock* /app/

# Install project dependencies (main only, no dev)
RUN uv sync

COPY . /app/

EXPOSE 7005

CMD ["uv", "run", ".", "--host", "0.0.0.0", "--port", "7005"]