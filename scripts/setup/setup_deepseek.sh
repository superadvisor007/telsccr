#!/bin/bash
# =============================================================================
# DeepSeek 7B Setup Script for TelegramSoccer
# 100% FREE LLM via Ollama - No API costs!
# =============================================================================

set -e

echo "🚀 Setting up DeepSeek 7B for TelegramSoccer..."
echo "================================================"
echo ""
echo "DeepSeek 7B is a powerful open-source LLM that runs locally."
echo "✅ No API keys needed"
echo "✅ No external costs"
echo "✅ Runs on GitHub Codespaces compute"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed successfully!"
else
    echo "✅ Ollama is already installed"
fi

# Start Ollama service
echo ""
echo "🔄 Starting Ollama service..."
ollama serve &
sleep 5

# Pull DeepSeek 7B model
echo ""
echo "📥 Pulling DeepSeek 7B model (this may take 5-10 minutes)..."
echo "   Model size: ~4.1GB"
echo ""

ollama pull deepseek-llm:7b

echo ""
echo "✅ DeepSeek 7B model downloaded successfully!"

# Also pull the coder variant as fallback
echo ""
echo "📥 Pulling DeepSeek Coder 7B (fallback model)..."
ollama pull deepseek-coder:7b

echo ""
echo "================================================"
echo "🎉 DeepSeek Setup Complete!"
echo "================================================"
echo ""
echo "Available models:"
ollama list
echo ""
echo "To test DeepSeek, run:"
echo "  python -c \"from src.llm import get_deepseek_llm; llm = get_deepseek_llm(); print(llm.is_available())\""
echo ""
echo "Environment variables (optional):"
echo "  LLM_MODEL=deepseek-llm:7b"
echo "  OLLAMA_HOST=http://localhost:11434"
echo ""
