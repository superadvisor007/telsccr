#!/bin/bash
# Disk Space Cleanup Script for Zero-Cost System

echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║     🧹 DISK SPACE CLEANUP & OPTIMIZATION 🧹            ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Show current disk usage
echo "📊 Current Disk Usage:"
df -h / | tail -1
echo ""

echo "🔍 Finding space hogs..."
echo ""

# 1. Clean pip cache
echo "[1/8] Cleaning pip cache..."
pip cache purge 2>/dev/null && echo "  ✓ Pip cache cleaned" || echo "  ℹ️  No pip cache to clean"

# 2. Clean apt cache
echo "[2/8] Cleaning apt cache..."
sudo apt-get clean 2>/dev/null && echo "  ✓ Apt cache cleaned" || echo "  ℹ️  Apt cache minimal"

# 3. Remove unnecessary packages
echo "[3/8] Removing unnecessary packages..."
sudo apt-get autoremove -y 2>/dev/null && echo "  ✓ Packages removed" || echo "  ℹ️  No packages to remove"

# 4. Clean /tmp
echo "[4/8] Cleaning /tmp..."
sudo find /tmp -type f -atime +2 -delete 2>/dev/null && echo "  ✓ Old tmp files removed" || echo "  ℹ️  Tmp clean"

# 5. Clean npm cache (if exists)
echo "[5/8] Cleaning npm cache..."
npm cache clean --force 2>/dev/null && echo "  ✓ Npm cache cleaned" || echo "  ℹ️  No npm cache"

# 6. Clean Docker (if running)
echo "[6/8] Cleaning Docker..."
docker system prune -af 2>/dev/null && echo "  ✓ Docker cleaned" || echo "  ℹ️  Docker not used"

# 7. Remove Python __pycache__
echo "[7/8] Removing Python cache..."
find /workspaces/telegramsoccer -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null && echo "  ✓ Python cache removed" || echo "  ℹ️  No cache found"
find /workspaces/telegramsoccer -type f -name "*.pyc" -delete 2>/dev/null && echo "  ✓ .pyc files removed" || echo "  ℹ️  No .pyc files"

# 8. Clean logs
echo "[8/8] Truncating large logs..."
find /workspaces/telegramsoccer/logs -type f -size +10M -exec truncate -s 0 {} \; 2>/dev/null && echo "  ✓ Large logs truncated" || echo "  ℹ️  No large logs"

echo ""
echo "✅ CLEANUP COMPLETE!"
echo ""
echo "📊 New Disk Usage:"
df -h / | tail -1
echo ""

# Calculate freed space
echo "🎯 Space Analysis:"
du -sh ~/.ollama/models 2>/dev/null | awk '{print "  • Ollama models: " $1}'
du -sh /workspaces/telegramsoccer/venv 2>/dev/null | awk '{print "  • Python venv: " $1}'
du -sh /workspaces/telegramsoccer/data 2>/dev/null | awk '{print "  • Data directory: " $1}'
echo ""

# Show top space consumers
echo "📈 Top 5 Space Consumers:"
du -sh /workspaces/telegramsoccer/* 2>/dev/null | sort -h | tail -5
echo ""

echo "💡 OPTIMIZATION TIPS:"
echo "  1. Use quantized Ollama model: ollama pull llama3.2:3b-q4_K_M (50% smaller)"
echo "  2. Remove unused models: ollama rm <model_name>"
echo "  3. Clear old data: rm -rf data/processed/* data/raw/*"
echo ""
echo "🎉 System optimized for zero-cost operations!"
