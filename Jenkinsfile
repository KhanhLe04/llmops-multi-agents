pipeline {
  agent none

  stages {

    stage('Test orchestrator-agent') {
      agent {
        kubernetes {
          yaml """
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: python
    image: python:3.13-slim
    command:
    - cat
    tty: true
"""
        }
      }
      steps {
        container('python') {
          dir('src/agents/orchestrator-agent') {
            sh """
              pip install --no-cache-dir uv==0.9.3
              uv pip install --no-cache-dir -r pyproject.toml
              
              echo "⚡ Running tests..."
              uv run pytest -q || exit 1
            """
          }
        }
      }
    }

    stage('Test rag-agent') {
      agent {
        kubernetes {
          yaml """
    apiVersion: v1
    kind: Pod
    spec:
    containers:
    - name: python
        image: python:3.13-slim
        command:
        - cat
        tty: true
    """
        }
      }
      steps {
        container('python') {
          dir('src/agents/rag-agent') {
            sh """
              pip install --no-cache-dir uv==0.9.3
              uv venv
              source .venv/bin/activate
              uv pip install --no-cache-dir -r pyproject.toml

              echo "⚡ Running tests..."
              uv run pytest -q || exit 1
            """
          }
        }
      }
    }
  }
}
