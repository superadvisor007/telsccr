#!/bin/bash
# Production Deployment Script - One Command Deploy
# Usage: ./deploy.sh [staging|production]

set -e  # Exit on error

ENVIRONMENT=${1:-staging}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 TelegramSoccer Deployment Script"
echo "=================================="
echo "Environment: $ENVIRONMENT"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found"
    exit 1
fi

if ! command -v git &> /dev/null; then
    print_error "Git not found"
    exit 1
fi

print_status "Prerequisites check passed"

# Create directories
echo ""
echo "📁 Creating directory structure..."
mkdir -p data/{historical,predictions,training,backtest_results}
mkdir -p models/knowledge_enhanced
mkdir -p logs
print_status "Directories created"

# Environment setup
echo ""
echo "🔧 Setting up Python environment..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1
print_status "Pip upgraded"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt
print_status "Dependencies installed"

# Check for trained models
echo ""
echo "🤖 Checking ML models..."

if [ ! -f "models/knowledge_enhanced/over_1_5_model.pkl" ]; then
    print_warning "No trained models found"
    echo "   Run: python train_knowledge_enhanced_ml.py"
else
    print_status "Trained models found"
fi

# Check environment variables
echo ""
echo "🔐 Checking environment variables..."

if [ ! -f ".env" ]; then
    print_warning "No .env file found"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_warning "Created .env from template - PLEASE UPDATE WITH YOUR KEYS"
    fi
else
    # Check for critical keys
    if grep -q "TELEGRAM_BOT_TOKEN=" .env && grep -q "TELEGRAM_CHAT_ID=" .env; then
        print_status "Environment variables configured"
    else
        print_error "Missing critical environment variables in .env"
        exit 1
    fi
fi

# Run health checks
echo ""
echo "🏥 Running health checks..."

# Test imports
python3 << EOF
import sys
try:
    import pandas
    import numpy
    import sklearn
    print("✅ Core ML packages: OK")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
EOF

# Git status
echo ""
echo "📊 Repository status:"
git status --short

# Final instructions
echo ""
echo "=================================="
echo "🎉 Deployment preparation complete!"
echo ""
echo "Next steps:"
if [ "$ENVIRONMENT" == "production" ]; then
    echo "1. Verify .env has production credentials"
    echo "2. Train models: source .venv/bin/activate && python train_knowledge_enhanced_ml.py"
    echo "3. Start bot: python src/main.py"
    echo "4. Monitor logs: tail -f logs/*.log"
else
    echo "1. Train models: source .venv/bin/activate && python train_knowledge_enhanced_ml.py"
    echo "2. Test pipeline: python scripts/runners/generate_enhanced_predictions.py"
    echo "3. Start bot: python src/main.py"
fi
echo ""
echo "Documentation: docs/README_PROFESSIONAL.md"
echo "=================================="
