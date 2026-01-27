# TelegramSoccer Zero-Cost Dev Container
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        gnupg \
        lsb-release \
        git \
        vim \
        nano \
        htop \
        tree \
        jq \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI (for Docker-in-Docker)
RUN curl -fsSL https://get.docker.com -o get-docker.sh \
    && sh get-docker.sh \
    && rm get-docker.sh

# Create workspace directory
RUN mkdir -p /workspace && chown vscode:vscode /workspace

# Set working directory
WORKDIR /workspace

# Switch to vscode user
USER vscode

# Install Python development tools
RUN python -m pip install --user --upgrade pip setuptools wheel

# Set environment variables
ENV PATH="/home/vscode/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8501 11434 5000
