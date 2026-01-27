#!/usr/bin/env python3
"""Quick Telegram Bot Test - sendet 5 Nachrichten"""
import asyncio
import httpx

TOKEN = "7971161852:AAFJAdHNAxYTHs2mi7Wj5sWuSA2tfA9WwcI"
BASE_URL = f"https://api.telegram.org/bot{TOKEN}"

async def test_bot():
    print("\n🤖 TELEGRAM BOT TEST")
    print("="*60 + "\n")
    
    async with httpx.AsyncClient() as client:
        # 1. Bot-Info
        print("1️⃣  Bot validieren...")
        r = await client.get(f"{BASE_URL}/getMe")
        if r.status_code != 200:
            print(f"   ❌ Fehler: {r.status_code}")
            return
        me = r.json()["result"]
        print(f"   ✅ @{me['username']} (ID: {me['id']})\n")
        
        # 2. Chat-ID finden
        print("2️⃣  Chat suchen...")
        r = await client.get(f"{BASE_URL}/getUpdates")
        updates = r.json().get("result", [])
        
        if not updates:
            print("   ⚠️  Keine Chats gefunden!\n")
            print("   📱 BITTE ERST:")
            print("      1. Öffne Telegram")
            print("      2. Suche @Tonticketbot")
            print("      3. Sende /start\n")
            return
        
        chat_id = updates[-1]["message"]["chat"]["id"]
        user = updates[-1]["message"]["from"].get("username", "Unbekannt")
        print(f"   ✅ Chat: {user} (ID: {chat_id})\n")
        
        # 3. Nachrichten senden
        print("3️⃣  Sende 5 Test-Nachrichten...\n")
        
        messages = [
            "🎯 *TEST 1* - Einfache Nachricht",
            "⚽ *TEST 2* - Emojis funktionieren! 🎉⚽🏆",
            "📊 *TEST 3* - Formatierung:\n• Punkt 1\n• Punkt 2\n• Punkt 3",
            "💰 *TEST 4* - Wett-Tipp Simulation:\n\n*Arsenal vs Manchester United*\n• Quote: 1.25\n• Wahrscheinlichkeit: 88%\n• Market: Over 1.5 Goals",
            "✅ *TEST 5* - Alle Tests erfolgreich!\n\n🎉 Telegram Bot funktioniert 100%!"
        ]
        
        for i, msg in enumerate(messages, 1):
            r = await client.post(
                f"{BASE_URL}/sendMessage",
                json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
            )
            if r.status_code == 200:
                print(f"   ✅ Nachricht {i}/5 gesendet")
            else:
                print(f"   ❌ Nachricht {i} fehlgeschlagen: {r.status_code}")
            await asyncio.sleep(0.5)
        
        print("\n" + "="*60)
        print("✅ TEST ABGESCHLOSSEN!")
        print("="*60)
        print("\n📱 Prüfe dein Telegram - du solltest 5 Nachrichten sehen!\n")

if __name__ == "__main__":
    asyncio.run(test_bot())
