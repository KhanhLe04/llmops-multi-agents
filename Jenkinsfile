pipeline {
    agent none

    environment {
        REGISTRY = "docker.io"
        ORCH_IMAGE = ""
        RAG_IMAGE  = ""
        IMAGE_TAG  = ""
        TAG_FROM_GIT = ""
        DOCKER_USERNAME = ""
    }

    stages {
        stage('Checkout') {
            agent any
            steps {
                checkout scm
                script {
                    env.IMAGE_TAG = sh(returnStdout: true, script: "git rev-parse --short HEAD").trim()
                    env.TAG_FROM_GIT = sh(returnStdout: true, script: "git describe --tags --exact-match || true").trim()
                    withCredentials([usernamePassword(credentialsId: 'docker-registry-creds', usernameVariable: 'REG_USERNAME', passwordVariable: 'REG_PASSWORD')]) {
                        env.DOCKER_USERNAME = REG_USERNAME
                        env.ORCH_IMAGE = "${REG_USERNAME}/orchestrator-agent"
                        env.RAG_IMAGE  = "${REG_USERNAME}/rag-agent"
                    }
                }
            }
        }

        stage('Orchestrator CI') {
            agent {
                docker {
                    image 'python:3.11'
                    args '--privileged -v /var/run/docker.sock:/var/run/docker.sock'
                }
            }
            steps {
                sh '''
                    set -euo pipefail
                    export DEBIAN_FRONTEND=noninteractive
                    apt-get update >/dev/null
                    apt-get install -y --no-install-recommends docker.io curl git >/dev/null
                    rm -rf /var/lib/apt/lists/*
                    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
                    export PATH="$HOME/.local/bin:$PATH"
                    cd src/agents/orchestrator-agent
                    uv sync
                    uv run pytest
                    docker build -f Dockerfile -t ${ORCH_IMAGE}:${IMAGE_TAG} .
                '''
            }
        }

        stage('RAG Agent CI') {
            agent {
                docker {
                    image 'python:3.11'
                    args '--privileged -v /var/run/docker.sock:/var/run/docker.sock'
                }
            }
            steps {
                sh '''
                    set -euo pipefail
                    export DEBIAN_FRONTEND=noninteractive
                    apt-get update >/dev/null
                    apt-get install -y --no-install-recommends docker.io curl git >/dev/null
                    rm -rf /var/lib/apt/lists/*
                    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
                    export PATH="$HOME/.local/bin:$PATH"
                    cd src/agents/rag-agent
                    uv sync
                    uv run pytest
                    docker build -f Dockerfile -t ${RAG_IMAGE}:${IMAGE_TAG} .
                '''
            }
        }

        stage('Tag & Push (on git tag)') {
            when {
                expression { return env.TAG_FROM_GIT?.trim() }
            }
            agent {
                docker {
                    image 'docker:24.0.7'
                    args '--privileged -v /var/run/docker.sock:/var/run/docker.sock'
                }
            }
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'docker-registry-creds', usernameVariable: 'REG_USERNAME', passwordVariable: 'REG_PASSWORD')]) {
                        sh """
                            echo ${REG_PASSWORD} | docker login -u ${REG_USERNAME} --password-stdin ${REGISTRY}
                            docker tag ${ORCH_IMAGE}:${IMAGE_TAG} ${ORCH_IMAGE}:${TAG_FROM_GIT}
                            docker tag ${RAG_IMAGE}:${IMAGE_TAG} ${RAG_IMAGE}:${TAG_FROM_GIT}
                            docker push ${ORCH_IMAGE}:${TAG_FROM_GIT}
                            docker push ${RAG_IMAGE}:${TAG_FROM_GIT}
                        """
                    }
                }
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}
