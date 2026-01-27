#!/bin/bash
# Post-start script: Runs every time container starts

set -e

echo "🚀 TelegramSoccer - Starting services..."
echo ""

# 1. Start Ollama
echo "1️⃣  Starting Ollama..."
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
    echo "   ✅ Ollama started (PID: $(pgrep -x ollama))"
else
    echo "   ✅ Ollama already running (PID: $(pgrep -x ollama))"
fi

# 2. Pull LLM model if not exists
echo ""
echo "2️⃣  Checking LLM model..."
if ! ollama list | grep -q "llama3.2:3b"; then
    echo "   ⏳ Downloading Llama 3.2 3B (2GB, ~2 minutes)..."
    ollama pull llama3.2:3b
    echo "   ✅ Model downloaded"
else
    echo "   ✅ Model already available"
fi

# 3. Activate venv
echo ""
echo "3️⃣  Activating Python environment..."
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
    echo "   ✅ venv activated"
fi

# 4. Show system status
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 SYSTEM STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "• Ollama:  $(pgrep -x ollama > /dev/null && echo '✅ Running' || echo '❌ Not running')"
echo "• Python:  $(python --version 2>&1)"
echo "• Disk:    $(df -h / | tail -1 | awk '{print $5 " used (" $3 "/" $2 ")"}')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Ready to code! Try:"
echo "   python test_truly_free_apis.py"
echo ""
