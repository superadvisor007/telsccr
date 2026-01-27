# 🔐 GitHub Secrets Setup Guide

Um das Elite Value Bets System automatisch täglich laufen zu lassen, musst du folgende Secrets in deinem GitHub Repository hinzufügen.

## 📍 Wo füge ich Secrets hinzu?

1. Gehe zu: https://github.com/superadvisor007/telegramsoccer/settings/secrets/actions
2. Klicke auf **"New repository secret"**

## 🔑 Erforderliche Secrets

### 1. TELEGRAM_BOT_TOKEN (BEREITS BEKANNT)
```
Name:  TELEGRAM_BOT_TOKEN
Value: 7971161852:AAFJAdHNAxYTHs2mi7Wj5sWuSA2tfA9WwcI
```

### 2. TELEGRAM_CHAT_ID (BEREITS BEKANNT)
```
Name:  TELEGRAM_CHAT_ID
Value: 7554175657
```

### 3. ODDS_API_KEY (KOSTENLOS HOLEN)

**So bekommst du einen kostenlosen API Key:**

1. Gehe zu: https://the-odds-api.com/
2. Klicke auf **"Get API Key"** (oben rechts)
3. Registriere dich mit Email
4. Du bekommst **500 kostenlose Requests pro Monat**
5. Kopiere deinen API Key

```
Name:  ODDS_API_KEY
Value: [dein-api-key-hier]
```

## ✅ Nach dem Setup

Die GitHub Actions werden automatisch:
- **Täglich um 8:00 UTC**: Elite Value Bets generieren und zu Telegram senden
- **Sonntags**: Self-Training Status prüfen

Du kannst auch manuell auslösen:
1. Gehe zu: https://github.com/superadvisor007/telegramsoccer/actions
2. Wähle "🏆 Daily Elite Value Bets"
3. Klicke "Run workflow"

## 🎯 Was passiert ohne ODDS_API_KEY?

Das System funktioniert trotzdem! Es berechnet dann realistische Odds basierend auf:
- Elo-Ratings der Teams
- Historische Durchschnittswerte der Liga
- Mathematische Modelle

Mit dem API Key bekommst du **echte Live-Odds** von Bookmakers wie:
- Bet365, Pinnacle, Betfair, etc.

## 📊 Kostenübersicht

| Service | Kosten | Limit |
|---------|--------|-------|
| GitHub Actions | GRATIS | 2000 Min/Monat |
| The-Odds-API | GRATIS | 500 Requests/Monat |
| Telegram Bot | GRATIS | Unbegrenzt |
| OpenLigaDB | GRATIS | Unbegrenzt |
| TheSportsDB | GRATIS | Unbegrenzt |

**Gesamtkosten: 0€/Monat** 🎉
