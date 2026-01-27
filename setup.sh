#!/bin/bash
# Setup script for TelegramSoccer

set -e

echo "🎯 TelegramSoccer Setup Script"
echo "=============================="

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version+ required. Found: $python_version"
    exit 1
fi
echo "✅ Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
pip install -r requirements-dev.txt --quiet
echo "✅ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/raw data/processed logs models
echo "✅ Directories created"

# Copy environment template
echo ""
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ .env file created from template"
    echo "⚠️  Please edit .env with your API keys!"
else
    echo "⚠️  .env file already exists"
fi

# Initialize database
echo ""
echo "Initializing database..."
python -c "from src.core.database import init_db; init_db()" 2>/dev/null && echo "✅ Database initialized" || echo "⚠️  Database initialization skipped (configure DATABASE_URL first)"

# Run tests
echo ""
echo "Running tests..."
pytest tests/ -q 2>/dev/null && echo "✅ Tests passed" || echo "⚠️  Some tests failed (expected if API keys not configured)"

echo ""
echo "=============================="
echo "🎉 Setup Complete!"
echo "=============================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Configure config/config.yaml as needed"
echo "3. Run: source venv/bin/activate"
echo "4. Test pipeline: python src/pipeline.py"
echo "5. Start bot: python src/main.py"
echo ""
echo "For Docker deployment:"
echo "  docker-compose up -d"
echo ""
echo "Read CONTRIBUTING.md for development guidelines."
echo ""
