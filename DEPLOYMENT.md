# Deployment Guide

## Quick Start

### 1. Local Development Setup

```bash
# Clone repository
git clone https://github.com/superadvisor007/telegramsoccer.git
cd telegramsoccer

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your credentials

# Create directories
mkdir -p data logs

# Run tests
python test.py

# Start bot
python run.py bot
```

### 2. Getting API Keys

#### Telegram Bot Token
1. Open [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` and follow instructions
3. Copy the token
4. Create a channel and add your bot as admin
5. Get channel ID (use @username or chat ID)

#### Stripe Setup
1. Go to [Stripe Dashboard](https://dashboard.stripe.com)
2. Get your API keys from Developers → API keys
3. Create products and prices:
   - Basic tier: CHF 9.90/month
   - Premium tier: CHF 19.90/month
4. Note the Price IDs (price_xxx)
5. Setup webhook endpoint (later)

#### API-Football
1. Register at [API-Football](https://www.api-football.com/)
2. Subscribe to a plan (free tier available)
3. Get your API key from dashboard

#### Groq API
1. Register at [Groq](https://console.groq.com/)
2. Get your API key
3. Free tier includes generous limits

### 3. Configuration

Edit `.env` file:

```bash
# Required
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHANNEL_ID=@your_channel
STRIPE_API_KEY=sk_test_xxxxxxxxxx
API_FOOTBALL_KEY=your_api_key
GROQ_API_KEY=gsk_xxxxxxxxxxxxxx

# Stripe Prices (create in Stripe dashboard)
STRIPE_PRICE_BASIC=price_xxxxx
STRIPE_PRICE_PREMIUM=price_xxxxx

# Admin users (comma-separated Telegram user IDs)
BOT_ADMIN_IDS=123456789,987654321
```

### 4. Running the Bot

#### Bot Mode (Telegram Bot)
```bash
python run.py bot
```

#### Webhook Mode (Payment Server)
```bash
python run.py webhook
```

#### Daily Tasks (Cron)
```bash
python run.py daily
```

## Production Deployment

### Option 1: Oracle Cloud (Free Tier)

Oracle Cloud offers always-free tier VMs perfect for this bot.

#### Setup VM

1. Create Oracle Cloud account
2. Create a Compute instance:
   - Shape: VM.Standard.E2.1.Micro (always free)
   - Image: Ubuntu 22.04
   - Configure networking (allow ports 80, 443, 8443)

3. SSH into VM:
```bash
ssh ubuntu@your-vm-ip
```

4. Install dependencies:
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv git nginx certbot python3-certbot-nginx
```

5. Clone and setup:
```bash
git clone https://github.com/superadvisor007/telegramsoccer.git
cd telegramsoccer
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

6. Configure environment:
```bash
cp .env.example .env
nano .env  # Edit with your credentials
```

#### Setup Services

Create systemd services:

**Bot Service** (`/etc/systemd/system/soccer-bot.service`):
```ini
[Unit]
Description=Swiss Soccer Tips Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/telegramsoccer
Environment=PATH=/home/ubuntu/telegramsoccer/venv/bin
ExecStart=/home/ubuntu/telegramsoccer/venv/bin/python run.py bot
Restart=always

[Install]
WantedBy=multi-user.target
```

**Webhook Service** (`/etc/systemd/system/soccer-webhook.service`):
```ini
[Unit]
Description=Swiss Soccer Tips Webhook Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/telegramsoccer
Environment=PATH=/home/ubuntu/telegramsoccer/venv/bin
ExecStart=/home/ubuntu/telegramsoccer/venv/bin/python run.py webhook
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start services:
```bash
sudo systemctl daemon-reload
sudo systemctl enable soccer-bot soccer-webhook
sudo systemctl start soccer-bot soccer-webhook
```

#### Setup Nginx

Configure Nginx (`/etc/nginx/sites-available/soccer-bot`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /webhook/stripe {
        proxy_pass http://localhost:8443/webhook/stripe;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /health {
        proxy_pass http://localhost:8443/health;
    }
}
```

Enable site and get SSL:
```bash
sudo ln -s /etc/nginx/sites-available/soccer-bot /etc/nginx/sites-enabled/
sudo certbot --nginx -d your-domain.com
sudo systemctl reload nginx
```

#### Configure Stripe Webhook

1. Go to Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://your-domain.com/webhook/stripe`
3. Select events:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`
4. Copy webhook secret to `.env`

### Option 2: GitHub Actions Only

For minimal setup, use GitHub Actions for daily predictions only (no 24/7 bot):

1. **Setup GitHub Secrets** (Settings → Secrets → Actions):
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHANNEL_ID`
   - `STRIPE_API_KEY`
   - `STRIPE_WEBHOOK_SECRET`
   - `STRIPE_PRICE_BASIC`
   - `STRIPE_PRICE_PREMIUM`
   - `API_FOOTBALL_KEY`
   - `GROQ_API_KEY`
   - `BOT_ADMIN_IDS`

2. **Workflow runs automatically** at 6 AM UTC daily

3. **For interactive bot**, still need a server (Option 1)

### Option 3: Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run.py", "bot"]
```

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  bot:
    build: .
    command: python run.py bot
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  webhook:
    build: .
    command: python run.py webhook
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    ports:
      - "8443:8443"
    restart: unless-stopped
```

Run:
```bash
docker-compose up -d
```

## Monitoring

### Check Service Status
```bash
sudo systemctl status soccer-bot
sudo systemctl status soccer-webhook
```

### View Logs
```bash
# Real-time logs
tail -f logs/bot.log

# Systemd logs
sudo journalctl -u soccer-bot -f
sudo journalctl -u soccer-webhook -f
```

### Test Endpoints
```bash
# Health check
curl http://localhost:8443/health

# Test webhook (requires valid signature)
curl -X POST http://localhost:8443/webhook/stripe
```

## Maintenance

### Update Code
```bash
cd /home/ubuntu/telegramsoccer
git pull
sudo systemctl restart soccer-bot soccer-webhook
```

### Backup Database
```bash
cp data/bot.db data/bot.db.backup.$(date +%Y%m%d)
```

### Check Disk Space
```bash
df -h
du -sh logs/
```

### Rotate Logs
Add to crontab:
```bash
0 0 * * 0 find /home/ubuntu/telegramsoccer/logs -name "*.log" -mtime +30 -delete
```

## Troubleshooting

### Bot Not Responding
```bash
# Check if running
sudo systemctl status soccer-bot

# Check logs
tail -100 logs/bot.log

# Restart
sudo systemctl restart soccer-bot
```

### Webhook Not Working
```bash
# Test connectivity
curl https://your-domain.com/health

# Check Stripe webhook logs in dashboard
# Verify webhook secret in .env
```

### Database Issues
```bash
# Check database file
ls -lh data/bot.db

# Verify database integrity
sqlite3 data/bot.db "PRAGMA integrity_check;"
```

### Out of Memory
```bash
# Check memory usage
free -h

# Adjust Python memory if needed (reduce workers, etc.)
```

## Security Checklist

- [ ] Use `.env` for all secrets (never commit)
- [ ] Enable firewall (ufw) on server
- [ ] Use HTTPS for webhooks (Let's Encrypt)
- [ ] Regularly update dependencies
- [ ] Monitor logs for suspicious activity
- [ ] Backup database regularly
- [ ] Use strong webhook secrets
- [ ] Limit admin user IDs

## Performance Tips

1. **Database**: Consider PostgreSQL for production scale
2. **Caching**: Add Redis for frequently accessed data
3. **Rate Limiting**: Implement rate limiting for API calls
4. **Load Balancing**: Use multiple instances behind load balancer
5. **Monitoring**: Add Prometheus + Grafana for metrics

## Cost Estimate

### Free Tier (Minimal)
- Oracle Cloud: Free forever
- GitHub Actions: Free (2000 minutes/month)
- Groq API: Free tier available
- API-Football: Free tier (100 requests/day)
- **Total: $0/month**

### Paid Tier (Recommended)
- API-Football Basic: ~$10/month
- Groq API: Pay-as-you-go (~$5/month)
- Domain: ~$12/year
- **Total: ~$15/month + domain**

Revenue from subscriptions will cover costs!

## Support

For issues or questions:
- GitHub Issues: https://github.com/superadvisor007/telegramsoccer/issues
- Email: your-support-email

## Next Steps

After deployment:
1. Test bot with `/start` command
2. Test subscription flow
3. Verify daily predictions are posted
4. Monitor for first few days
5. Promote to users!

Good luck! 🚀⚽
