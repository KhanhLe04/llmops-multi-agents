pipeline {
    agent any

    stages {

        stage('Build orchestrator-agent') {
            agent {
                docker {
                    image 'python:3.13-slim'
                    args '-u root'
                }
            }
            steps {
                dir('src/agents/orchestrator-agent') {
                    sh """
                        pip install --no-cache-dir uv==0.9.3
                        uv pip install --no-cache-dir -r pyproject.toml
                        uv run .
                    """
                }
            }
        }

        stage('Build rag-agent') {
            agent {
                docker {
                    image 'python:3.13-slim'
                    args '-u root'
                }
            }
            steps {
                dir('src/agents/rag-agent') {
                    sh """
                        pip install --no-cache-dir uv==0.9.3
                        uv pip install --no-cache-dir -r pyproject.toml
                        uv run .
                    """
                }
            }
        }

    }
}
