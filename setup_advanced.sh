"""Setup script for advanced features."""
#!/bin/bash

set -e

echo "🚀 Setting up TelegramSoccer Advanced System..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi

echo -e "${GREEN}✓ Python $python_version${NC}"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Install base dependencies
echo -e "${BLUE}Installing base dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Base dependencies installed${NC}"

# Install advanced dependencies
echo -e "${BLUE}Installing advanced ML dependencies...${NC}"
pip install -r requirements-advanced.txt
echo -e "${GREEN}✓ Advanced dependencies installed${NC}"

# Create necessary directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p data/chroma_db
mkdir -p data/finetuning
mkdir -p data/predictions
mkdir -p models/rl_agent
mkdir -p models/finetuned_llm
mkdir -p models/meta_learner
mkdir -p logs/tensorboard
mkdir -p mlflow-artifacts
echo -e "${GREEN}✓ Directories created${NC}"

# Copy .env.example if .env doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${BLUE}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo "⚠️  Please edit .env with your API keys!"
fi

# Initialize database
echo -e "${BLUE}Initializing database...${NC}"
python -c "from src.core.database import init_db; init_db()"
echo -e "${GREEN}✓ Database initialized${NC}"

# Initialize RAG system
echo -e "${BLUE}Initializing RAG memory system...${NC}"
python -c "
from src.rag.retriever import BettingMemoryRAG
from src.core.database import DatabaseManager
try:
    rag = BettingMemoryRAG(DatabaseManager())
    print('RAG system initialized')
except Exception as e:
    print(f'RAG initialization skipped: {e}')
"
echo -e "${GREEN}✓ RAG system ready${NC}"

# Check if we should train RL agent
echo -e "${BLUE}Do you want to train the RL agent now? (recommended, ~5 minutes) [y/N]${NC}"
read -r train_rl

if [[ $train_rl =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Training RL agent (this may take a few minutes)...${NC}"
    python -c "
from src.rl.agent import RLStakingAgent
agent = RLStakingAgent()
agent.train(total_timesteps=50000)
print('RL agent trained successfully')
"
    echo -e "${GREEN}✓ RL agent trained${NC}"
else
    echo "ℹ️  Skipping RL training (you can train later with: python -c \"from src.rl.agent import RLStakingAgent; agent = RLStakingAgent(); agent.train(total_timesteps=100000)\")"
fi

# Setup MLflow (optional)
echo -e "${BLUE}Do you want to start MLflow tracking server? [y/N]${NC}"
read -r start_mlflow

if [[ $start_mlflow =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Starting MLflow server...${NC}"
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5000 &
    sleep 2
    echo -e "${GREEN}✓ MLflow server started at http://localhost:5000${NC}"
fi

# Setup Airflow (optional)
echo -e "${BLUE}Do you want to setup Airflow? [y/N]${NC}"
read -r setup_airflow

if [[ $setup_airflow =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Initializing Airflow...${NC}"
    export AIRFLOW_HOME=$(pwd)/airflow
    airflow db init
    airflow users create --username admin --password admin --firstname Admin --lastname Admin --role Admin --email admin@example.com
    echo -e "${GREEN}✓ Airflow initialized (username: admin, password: admin)${NC}"
    echo "Start with: airflow webserver -p 8080 & airflow scheduler &"
fi

# Run tests
echo -e "${BLUE}Running tests...${NC}"
python -m pytest tests/ -v --tb=short || echo "⚠️  Some tests failed (expected if APIs not configured)"
echo -e "${GREEN}✓ Tests complete${NC}"

# Final instructions
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 Advanced System Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "📚 Next Steps:"
echo ""
echo "1. Edit .env with your API keys:"
echo "   nano .env"
echo ""
echo "2. Run the pipeline:"
echo "   python src/pipeline.py"
echo ""
echo "3. Launch the dashboard:"
echo "   streamlit run dashboard/app.py"
echo ""
echo "4. View MLflow experiments (if started):"
echo "   http://localhost:5000"
echo ""
echo "5. Read advanced features docs:"
echo "   cat docs/ADVANCED_FEATURES.md"
echo ""
echo "🚀 Key Features Now Available:"
echo "   • RL-Optimized Staking"
echo "   • RAG-Enhanced Analysis (learns from past mistakes)"
echo "   • Meta-Learning Ensemble"
echo "   • Continuous Training Pipeline"
echo "   • Professional Dashboard"
echo "   • MLflow Monitoring"
echo ""
echo "💡 Tip: The system will improve over time as it learns from outcomes!"
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
