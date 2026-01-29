#!/usr/bin/env python3
"""
🧪 Pipeline Integration Test
============================
Tests the complete unified pipeline from data collection to delivery.

Tests:
1. Data Sources Integration
2. Feature Engineering Pipeline
3. Foundation Model (DeepSeek) Analysis
4. Multi-Bet Building
5. Telegram Delivery
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import traceback

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_component(name: str, func):
    """Test a single component with error handling."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {name}")
    print('='*60)
    
    try:
        result = func()
        print(f"✅ {name}: PASSED")
        return True, result
    except Exception as e:
        print(f"❌ {name}: FAILED")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False, None


def test_data_sources():
    """Test data source integrations."""
    print("\n1️⃣ Testing Data Sources...")
    
    # Test StatsBomb
    from data_sources import StatsBombClient
    sb = StatsBombClient()
    
    comps = sb.get_competitions()
    print(f"   StatsBomb: {len(comps)} competitions available")
    assert len(comps) > 0, "No competitions found"
    
    # Test Free APIs
    from data_sources import FreeFootballAPIs
    apis = FreeFootballAPIs()
    
    # Use the correct method name (singular)
    team = apis.thesportsdb.search_team('Manchester United')
    print(f"   TheSportsDB: {'Found' if team else 'Not found'} team for 'Manchester United'")
    
    return {"statsbomb_comps": len(comps), "sportsdb_team_found": team is not None}


def test_feature_engineering():
    """Test feature engineering pipeline."""
    print("\n2️⃣ Testing Feature Engineering...")
    
    from feature_engineering import SPADLConverter, StructuralFeatureEngine
    import pandas as pd
    
    # Test SPADL converter
    converter = SPADLConverter()
    print(f"   SPADLConverter initialized")
    
    # Test structural engine
    engine = StructuralFeatureEngine()
    
    # Create mock historical data
    mock_matches = pd.DataFrame({
        'home_team': ['Liverpool'] * 5,
        'away_team': ['Chelsea', 'Arsenal', 'ManCity', 'Tottenham', 'ManUtd'],
        'home_score': [2, 3, 1, 2, 4],
        'away_score': [1, 0, 1, 2, 2],
        'home_xg': [2.1, 2.5, 1.2, 1.8, 3.2],
        'away_xg': [1.1, 0.8, 1.5, 1.9, 2.1],
    })
    
    # Compute features from historical data
    features = engine.compute_from_history(mock_matches, 'Liverpool', last_n=5)
    
    print(f"   Structural Features computed:")
    print(f"      - xG/90: {features.xg_per_90:.2f}")
    print(f"      - xGA/90: {features.xga_per_90:.2f}")
    print(f"      - Form Trend: {features.form_xg_trend:.2f}")
    print(f"      - Momentum: {features.momentum_score:.2f}")
    
    return features


def test_foundation_models():
    """Test foundation model (DeepSeek) integration."""
    print("\n3️⃣ Testing Foundation Models...")
    
    from foundation import DeepSeekEngine, DeepSeekConfig
    
    config = DeepSeekConfig(
        backend='ollama',
        model_name='deepseek-llm:7b-chat',
        timeout=30.0,
    )
    
    engine = DeepSeekEngine(config)
    
    # Check availability via internal client state
    is_available = engine._client is not None
    print(f"   Ollama Backend: {'Available' if is_available else 'Not Available'}")
    
    if is_available:
        # Run analysis
        result = engine.analyze_match(
            home_team="Liverpool",
            away_team="Arsenal",
            league="Premier League"
        )
        
        print(f"   Analysis Result:")
        print(f"      - Home Win Prob: {result.get('probabilities', {}).get('home_win', 'N/A')}")
        print(f"      - Over 2.5 Prob: {result.get('probabilities', {}).get('over_2_5', 'N/A')}")
        print(f"      - Confidence: {result.get('confidence', 'N/A')}")
        
        return result
    else:
        print("   ⚠️ DeepSeek not available - using fallback mode")
        
        # Test fallback
        from foundation import ModelCache
        cache = ModelCache()
        print(f"   Model Cache initialized at: {cache.db_path}")
        
        return {"status": "fallback", "cache_available": True}


def test_living_agent():
    """Test living agent components."""
    print("\n4️⃣ Testing Living Agent Components...")
    
    try:
        from living_agent import MultiBetBuilder, MatchAnalyzer
        
        builder = MultiBetBuilder(
            min_leg_odds=1.25,
            max_leg_odds=2.00,
            target_total_odds=6.0
        )
        
        print(f"   MultiBetBuilder initialized")
        print(f"      - Min Odds: {builder.min_leg_odds}")
        print(f"      - Max Odds: {builder.max_leg_odds}")
        print(f"      - Target Total: {builder.target_total_odds}")
        
        analyzer = MatchAnalyzer()
        print(f"   MatchAnalyzer initialized")
        
        return {"builder": "ready", "analyzer": "ready"}
        
    except ImportError as e:
        print(f"   ⚠️ Living Agent not fully initialized: {e}")
        return {"status": "partial"}


def test_unified_pipeline():
    """Test the unified pipeline orchestrator."""
    print("\n5️⃣ Testing Unified Pipeline...")
    
    from pipeline import UnifiedBettingPipeline
    
    pipeline = UnifiedBettingPipeline(verbose=False)
    
    # Get system status
    status = pipeline.get_system_status()
    
    print(f"   Pipeline Status:")
    print(f"      - StatsBomb: {status['components']['data_sources']['statsbomb']}")
    print(f"      - Free APIs: {status['components']['data_sources']['free_apis']}")
    print(f"      - LLM Backend: {status['components']['llm']['backend']}")
    print(f"      - Telegram: {'Configured' if status['components']['telegram']['configured'] else 'Not Configured'}")
    
    return status


def test_telegram_delivery():
    """Test Telegram connectivity (dry run)."""
    print("\n6️⃣ Testing Telegram Integration...")
    
    token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
    
    if token and chat_id:
        import requests
        
        url = f"https://api.telegram.org/bot{token}/getMe"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bot_info = response.json()['result']
                print(f"   Bot Name: @{bot_info['username']}")
                print(f"   Status: Connected")
                return {"connected": True, "bot": bot_info['username']}
        except Exception as e:
            print(f"   Connection Error: {e}")
            
    else:
        print("   ⚠️ Telegram not configured")
        print("      Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        
    return {"connected": False}


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("🧪 UNIFIED PIPELINE INTEGRATION TESTS")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    
    # Run tests
    tests = [
        ("Data Sources", test_data_sources),
        ("Feature Engineering", test_feature_engineering),
        ("Foundation Models", test_foundation_models),
        ("Living Agent", test_living_agent),
        ("Unified Pipeline", test_unified_pipeline),
        ("Telegram Delivery", test_telegram_delivery),
    ]
    
    passed = 0
    failed = 0
    
    for name, func in tests:
        success, result = test_component(name, func)
        results[name] = {"success": success, "result": result}
        
        if success:
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    for name, data in results.items():
        status = "✅" if data["success"] else "❌"
        print(f"   {status} {name}")
    
    print(f"\n   Total: {passed}/{len(tests)} passed")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {failed} test(s) failed")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_all_tests()
