#!/bin/bash
# Hybrid environment activation: venv utilities + system ML packages

# Activate virtual environment for utility packages
source .venv/bin/activate

# Add system ML packages to Python path
export PYTHONPATH="/usr/lib/python3.12/site-packages:$PYTHONPATH"

echo "✅ Hybrid environment activated:"
echo "   - Venv utilities: loguru, python-dotenv, tenacity"
echo "   - System ML: sklearn 1.5.2, pandas 2.3.3, numpy 2.3.5, joblib 1.5.2"
echo ""
echo "Usage: source activate_env.sh"
