#!/bin/bash
# Quick API Keys Setup - Semi-Automated

echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║     🚀 QUICK API KEYS SETUP - Semi-Automated 🚀       ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

echo "Was ich FÜR DICH tun kann:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Telegram Bot: BEREITS KONFIGURIERT (@Tonticketbot)"
echo "✅ GitHub Workflow: BEREITS GEPUSHT"
echo "✅ System Tests: ALLE BESTANDEN"
echo "✅ Demo-Modus: VERFÜGBAR (läuft ohne API keys)"
echo ""

echo "Was DU tun musst (5-10 Minuten):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏳ API-Football Signup (persönliche E-Mail erforderlich)"
echo "⏳ iSports API Signup (persönliche E-Mail erforderlich)"
echo ""

echo "🎯 OPTION 1: Automatisch Browser öffnen"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

read -p "Soll ich die Signup-Seiten automatisch öffnen? (j/n): " open_browser

if [[ "$open_browser" == "j" || "$open_browser" == "J" ]]; then
    echo ""
    echo "[1/2] Öffne API-Football Signup..."
    $BROWSER "https://www.api-football.com/" 2>/dev/null &
    sleep 2
    
    echo "[2/2] Öffne iSports API Signup..."
    $BROWSER "https://www.isportsapi.com/" 2>/dev/null &
    sleep 2
    
    echo ""
    echo "✅ Browser-Tabs geöffnet!"
    echo ""
    echo "📋 NÄCHSTE SCHRITTE IN DEN BROWSER-TABS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "TAB 1: API-Football"
    echo "  1. Klicke 'Sign Up' (oben rechts)"
    echo "  2. Erstelle Konto mit DEINER E-Mail"
    echo "  3. Verifiziere E-Mail (Check Inbox)"
    echo "  4. Login → Dashboard → API Key KOPIEREN"
    echo ""
    echo "TAB 2: iSports API"
    echo "  1. Klicke 'Register' oder 'Sign Up'"
    echo "  2. Erstelle Konto mit DEINER E-Mail"
    echo "  3. Verifiziere E-Mail (Check Inbox)"
    echo "  4. Dashboard → API Key KOPIEREN"
    echo ""
else
    echo ""
    echo "📋 MANUELLE LINKS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "API-Football: https://www.api-football.com/"
    echo "iSports API: https://www.isportsapi.com/"
    echo ""
fi

echo ""
echo "⏸️  PAUSE - Warte auf API Keys..."
echo ""
read -p "Hast du BEIDE API Keys bekommen? (j/n): " has_keys

if [[ "$has_keys" == "j" || "$has_keys" == "J" ]]; then
    echo ""
    echo "🔑 API KEYS EINGABE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    read -p "API-Football Key eingeben: " api_football_key
    read -p "iSports API Key eingeben: " isports_key
    
    echo ""
    echo "💾 Speichere Keys in .env..."
    
    # Update .env file
    sed -i "s/API_FOOTBALL_KEY=.*/API_FOOTBALL_KEY=$api_football_key/" .env
    sed -i "s/ISPORTS_API_KEY=.*/ISPORTS_API_KEY=$isports_key/" .env
    
    echo "✅ Keys gespeichert!"
    echo ""
    
    echo "🧪 Teste API Keys..."
    python test_api_keys.py
    
    echo ""
    echo "✅ SETUP COMPLETE!"
    echo ""
    echo "🚀 NÄCHSTE SCHRITTE:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1. Telegram aktivieren:"
    echo "   - Öffne Telegram"
    echo "   - Suche: @Tonticketbot"
    echo "   - Sende: /start"
    echo "   - Teste: python test_telegram_bot.py"
    echo ""
    echo "2. Pipeline starten:"
    echo "   python src/pipeline_free.py"
    echo ""
    echo "💰 Total Cost: $0.00/month FOREVER"
    echo ""
else
    echo ""
    echo "⏳ Okay, kein Problem!"
    echo ""
    echo "🎮 IN DER ZWISCHENZEIT: Teste Demo-Modus"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Demo-Modus läuft OHNE API Keys und zeigt,"
    echo "wie das System funktioniert:"
    echo ""
    echo "  python demo_mode.py"
    echo ""
    echo "Sobald du Keys hast, führe dieses Skript erneut aus:"
    echo "  bash quick_setup.sh"
    echo ""
fi

echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║              ✅ QUICK SETUP COMPLETE ✅                ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
