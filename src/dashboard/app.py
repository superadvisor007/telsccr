#!/usr/bin/env python3
"""Simple Web Dashboard - Battle Tested"""
from flask import Flask, render_template, jsonify
import json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

def load_metrics():
    """Load current metrics"""
    metrics_file = Path("data/monitoring/health_metrics.json")
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {}

@app.route('/')
def index():
    """Main dashboard"""
    metrics = load_metrics()
    return render_template('dashboard.html', metrics=metrics)

@app.route('/api/health')
def api_health():
    """Health API endpoint"""
    return jsonify(load_metrics())

@app.route('/api/cache')
def api_cache():
    """Cache statistics"""
    from src.core.simple_cache import _cache
    return jsonify(_cache.get_stats())

if __name__ == "__main__":
    print("🌐 Starting Dashboard on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
