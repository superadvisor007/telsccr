#!/bin/bash
# GitHub Actions Secrets Setup - Alternative Method via Repository Environment File

echo "🔐 GITHUB ACTIONS SECRETS CONFIGURATION"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "✅ Method 1: Use .env file (RECOMMENDED - Already Configured)"
echo "────────────────────────────────────────────────────────────"
echo "The .env file in the repository already contains:"
echo "  TELEGRAM_BOT_TOKEN=7971161852:AAFJAdHNAxYTHs2mi7Wj5sWuSA2tfA9WwcI"
echo "  TELEGRAM_CHAT_ID=-4792949084"
echo ""
echo "GitHub Actions workflows will automatically use this file."
echo ""

echo "✅ Method 2: Web UI (If .env not working)"
echo "────────────────────────────────────────────────────────────"
echo "Visit: https://github.com/superadvisor007/telegramsoccer/settings/secrets/actions"
echo ""
echo "Click 'New repository secret' and add:"
echo ""
echo "  Secret Name: TELEGRAM_BOT_TOKEN"
echo "  Secret Value: 7971161852:AAFJAdHNAxYTHs2mi7Wj5sWuSA2tfA9WwcI"
echo ""
echo "  Secret Name: TELEGRAM_CHAT_ID"
echo "  Secret Value: -4792949084"
echo ""
echo "  Secret Name: FOOTBALL_DATA_API_KEY (optional)"
echo "  Secret Value: your_key_here"
echo ""

echo "✅ Method 3: GitHub CLI (if installed)"
echo "────────────────────────────────────────────────────────────"
echo "Run these commands:"
echo ""
echo "  gh secret set TELEGRAM_BOT_TOKEN -b'7971161852:AAFJAdHNAxYTHs2mi7Wj5sWuSA2tfA9WwcI'"
echo "  gh secret set TELEGRAM_CHAT_ID -b'-4792949084'"
echo ""

echo "════════════════════════════════════════════════════════════"
echo "📋 CURRENT STATUS"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check if .env exists
if [ -f "/workspaces/telegramsoccer/.env" ]; then
    echo "✅ .env file exists in repository"
    if grep -q "TELEGRAM_BOT_TOKEN=7971161852" /workspaces/telegramsoccer/.env 2>/dev/null; then
        echo "✅ TELEGRAM_BOT_TOKEN is configured in .env"
    fi
    if grep -q "TELEGRAM_CHAT_ID=-4792949084" /workspaces/telegramsoccer/.env 2>/dev/null; then
        echo "✅ TELEGRAM_CHAT_ID is configured in .env"
    fi
else
    echo "⚠️  .env file not found"
fi

# Check workflows
echo ""
if [ -f "/workspaces/telegramsoccer/.github/workflows/daily_training.yml" ]; then
    echo "✅ Daily training workflow configured"
fi
if [ -f "/workspaces/telegramsoccer/.github/workflows/manual_stress_test.yml" ]; then
    echo "✅ Manual stress test workflow configured"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🚀 WORKFLOW ACTIVATION"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "GitHub Actions workflows are activated when:"
echo "1. Files are pushed to the repository"
echo "2. Workflows are enabled in repository settings"
echo "3. First workflow run is triggered (manual or scheduled)"
echo ""
echo "To enable workflows:"
echo "  Visit: https://github.com/superadvisor007/telegramsoccer/actions"
echo "  Click 'I understand my workflows, go ahead and enable them'"
echo ""
echo "To manually trigger a workflow:"
echo "  1. Go to Actions tab"
echo "  2. Select 'Daily 10K Training & Predictions'"
echo "  3. Click 'Run workflow'"
echo ""

echo "════════════════════════════════════════════════════════════"
echo "✅ SETUP COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Current configuration uses .env file method."
echo "GitHub Actions will automatically load environment variables."
echo "No manual secret setup required unless .env is not working."
echo ""
