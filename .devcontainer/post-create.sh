#!/bin/bash
# Post-create script: Runs once when container is created

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  TelegramSoccer Dev Container - Post Create Setup           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 1. Install Python dependencies
echo "1️⃣  Installing Python dependencies..."
if [ -f requirements-free.txt ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements-free.txt
    echo "   ✅ Python dependencies installed"
else
    echo "   ⚠️  requirements-free.txt not found"
fi

# 2. Install Ollama
echo ""
echo "2️⃣  Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "   ✅ Ollama installed"
else
    echo "   ✅ Ollama already installed"
fi

# 3. Create directories
echo ""
echo "3️⃣  Creating directories..."
mkdir -p data logs models data/chroma_db
touch data/.gitkeep logs/.gitkeep models/.gitkeep
echo "   ✅ Directories created"

# 4. Initialize database
echo ""
echo "4️⃣  Initializing database..."
if [ -f src/core/database.py ]; then
    source venv/bin/activate
    python -c "from src.core.database import init_db; init_db()" 2>/dev/null || echo "   ⚠️  Database init skipped"
    echo "   ✅ Database initialized"
fi

# 5. Setup git
echo ""
echo "5️⃣  Configuring git..."
git config --global --add safe.directory /workspace
echo "   ✅ Git configured"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Post-Create Setup Complete!                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Next steps:"
echo "   • Container will auto-start Ollama on startup"
echo "   • Run: python test_truly_free_apis.py"
echo "   • Run: python demo_mode.py"
echo ""
