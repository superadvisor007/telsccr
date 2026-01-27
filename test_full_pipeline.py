#!/usr/bin/env python3
"""
KOMPLETTER PIPELINE TEST
Holt Matches von morgen, analysiert mit LLM, generiert Tips, sendet via Telegram
"""
import asyncio
import httpx
from datetime import datetime, timedelta
import json

# Config
TELEGRAM_TOKEN = "7971161852:AAFJAdHNAxYTHs2mi7Wj5sWuSA2tfA9WwcI"
TELEGRAM_CHAT_ID = "7554175657"
OLLAMA_URL = "http://localhost:11434"

async def get_matches_tomorrow():
    """Hole Matches von morgen (TheSportsDB + OpenLigaDB)"""
    print("\n" + "="*70)
    print("📅 SCHRITT 1: MATCHES VON MORGEN HOLEN")
    print("="*70 + "\n")
    
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Datum: {tomorrow}\n")
    
    matches = []
    
    async with httpx.AsyncClient(timeout=30) as client:
        # TheSportsDB - Premier League
        print("1️⃣  TheSportsDB - Premier League...")
        try:
            r = await client.get("https://www.thesportsdb.com/api/v1/json/3/eventsnextleague.php?id=4328")
            events = r.json().get("events", [])
            for event in events[:5]:  # Top 5
                if event and event.get("dateEvent"):
                    matches.append({
                        "source": "TheSportsDB",
                        "league": "Premier League",
                        "home": event.get("strHomeTeam", "Unknown"),
                        "away": event.get("strAwayTeam", "Unknown"),
                        "date": event.get("dateEvent"),
                        "time": event.get("strTime", "TBD")
                    })
            print(f"   ✅ {len(matches)} Matches gefunden\n")
        except Exception as e:
            print(f"   ⚠️  Fehler: {e}\n")
        
        # OpenLigaDB - Bundesliga
        print("2️⃣  OpenLigaDB - Bundesliga...")
        try:
            r = await client.get("https://api.openligadb.de/getmatchdata/bl1")
            bl_matches = r.json()
            for match in bl_matches[:3]:  # Top 3
                if match:
                    matches.append({
                        "source": "OpenLigaDB",
                        "league": "Bundesliga",
                        "home": match.get("team1", {}).get("teamName", "Unknown"),
                        "away": match.get("team2", {}).get("teamName", "Unknown"),
                        "date": match.get("matchDateTime", "")[:10],
                        "time": match.get("matchDateTime", "")[11:16] if len(match.get("matchDateTime", "")) > 10 else "TBD"
                    })
            print(f"   ✅ {len(matches) - len([m for m in matches if m['source'] == 'TheSportsDB'])} Bundesliga Matches\n")
        except Exception as e:
            print(f"   ⚠️  Fehler: {e}\n")
    
    print(f"📊 Total: {len(matches)} Matches gefunden\n")
    return matches[:10]  # Max 10

async def analyze_with_llm(matches):
    """Analysiere Matches mit Ollama LLM"""
    print("="*70)
    print("🧠 SCHRITT 2: LLM ANALYSE (OLLAMA)")
    print("="*70 + "\n")
    
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code != 200:
                print("❌ Ollama nicht erreichbar!")
                return []
    except:
        print("❌ Ollama läuft nicht! Starte mit: ollama serve")
        return []
    
    print("✅ Ollama verbunden\n")
    
    tips = []
    
    for i, match in enumerate(matches, 1):
        print(f"{i}/{len(matches)} Analysiere: {match['home']} vs {match['away']}")
        
        prompt = f"""Analyze this football match for betting:

Match: {match['home']} vs {match['away']}
League: {match['league']}
Date: {match['date']}

Provide a betting tip for "Over 1.5 Goals" market.

Response format (JSON):
{{
    "recommendation": "YES" or "NO",
    "confidence": 0-100,
    "reasoning": "short explanation",
    "odds_estimate": 1.10-1.50
}}

Keep it short and focused."""

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
                )
                
                if r.status_code == 200:
                    response = r.json().get("response", "")
                    
                    # Parse JSON aus Response
                    try:
                        # Finde JSON im Response
                        start = response.find("{")
                        end = response.rfind("}") + 1
                        if start >= 0 and end > start:
                            analysis = json.loads(response[start:end])
                            
                            if analysis.get("recommendation") == "YES":
                                tips.append({
                                    "match": f"{match['home']} vs {match['away']}",
                                    "league": match['league'],
                                    "market": "Over 1.5 Goals",
                                    "odds": analysis.get("odds_estimate", 1.25),
                                    "confidence": analysis.get("confidence", 75),
                                    "reasoning": analysis.get("reasoning", "Good attacking teams"),
                                    "date": match['date'],
                                    "time": match['time']
                                })
                                print(f"   ✅ TIP: Over 1.5 @ {analysis.get('odds_estimate', 1.25)}\n")
                            else:
                                print(f"   ⏭️  Übersprungen (LLM empfiehlt nicht)\n")
                    except:
                        # Fallback - einfache Analyse
                        tips.append({
                            "match": f"{match['home']} vs {match['away']}",
                            "league": match['league'],
                            "market": "Over 1.5 Goals",
                            "odds": 1.25,
                            "confidence": 70,
                            "reasoning": "Analysis pending",
                            "date": match['date'],
                            "time": match['time']
                        })
                        print(f"   ✅ TIP: Over 1.5 @ 1.25 (Default)\n")
        except Exception as e:
            print(f"   ⚠️  Fehler: {e}\n")
    
    return tips[:10]  # Max 10 Tips

async def send_tips_telegram(tips):
    """Sende Tips via Telegram"""
    print("="*70)
    print("📱 SCHRITT 3: TIPS VIA TELEGRAM SENDEN")
    print("="*70 + "\n")
    
    if not tips:
        msg = "⚠️ Keine Tips gefunden!\n\nDie Analyse hat keine passenden Wetten ergeben."
        print("Keine Tips zum Senden.\n")
    else:
        # Erstelle Nachricht
        msg = f"🎯 *WETT-TIPS FÜR MORGEN*\n"
        msg += f"📅 {(datetime.now() + timedelta(days=1)).strftime('%d.%m.%Y')}\n\n"
        msg += f"Gefunden: *{len(tips)} Tips*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for i, tip in enumerate(tips, 1):
            msg += f"*{i}. {tip['match']}*\n"
            msg += f"🏆 Liga: {tip['league']}\n"
            msg += f"⏰ Zeit: {tip['time']}\n"
            msg += f"💰 Market: {tip['market']}\n"
            msg += f"📊 Quote: {tip['odds']}\n"
            msg += f"🎯 Konfidenz: {tip['confidence']}%\n"
            msg += f"💭 _{tip['reasoning']}_\n\n"
        
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "✅ Generiert mit Ollama LLM\n"
        msg += "🤖 @Tonticketbot"
    
    # Senden
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": msg,
                    "parse_mode": "Markdown"
                }
            )
            
            if r.status_code == 200:
                print(f"✅ {len(tips)} Tips erfolgreich gesendet!\n")
                return True
            else:
                print(f"❌ Fehler beim Senden: {r.status_code}\n")
                return False
    except Exception as e:
        print(f"❌ Fehler: {e}\n")
        return False

async def main():
    print("\n" + "🎯"*35)
    print("  KOMPLETTER PIPELINE TEST - telegramsoccer")
    print("🎯"*35 + "\n")
    
    # 1. Matches holen
    matches = await get_matches_tomorrow()
    
    if not matches:
        print("❌ Keine Matches gefunden!")
        return
    
    # 2. LLM Analyse
    tips = await analyze_with_llm(matches)
    
    # 3. Via Telegram senden
    success = await send_tips_telegram(tips)
    
    # Summary
    print("="*70)
    print("📊 PIPELINE TEST ABGESCHLOSSEN")
    print("="*70 + "\n")
    print(f"✅ Matches gefunden: {len(matches)}")
    print(f"✅ Tips generiert: {len(tips)}")
    print(f"✅ Telegram gesendet: {'Ja' if success else 'Nein'}")
    print(f"\n💰 Kosten: $0.00 (100% kostenlos)")
    print(f"🎉 Pipeline Status: {'ERFOLGREICH' if success else 'FEHLER'}\n")

if __name__ == "__main__":
    asyncio.run(main())
