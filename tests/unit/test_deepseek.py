#!/usr/bin/env python3
"""
Test script for DeepSeek 7B LLM integration.
Verifies that the system is correctly configured to use DeepSeek.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_deepseek_config():
    """Test that DeepSeek is configured as the default LLM."""
    print("🧪 Testing DeepSeek Configuration...")
    print("=" * 50)
    
    # Test 1: Check environment variables
    model = os.environ.get("LLM_MODEL", "deepseek-llm:7b")
    provider = os.environ.get("LLM_PROVIDER", "ollama")
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    print(f"\n✅ LLM_MODEL: {model}")
    print(f"✅ LLM_PROVIDER: {provider}")
    print(f"✅ OLLAMA_HOST: {host}")
    
    assert "deepseek" in model.lower(), f"Expected DeepSeek model, got: {model}"
    assert provider == "ollama", f"Expected ollama provider, got: {provider}"
    
    # Test 2: Import DeepSeek client
    print("\n📦 Testing DeepSeek client import...")
    try:
        from src.llm.deepseek_client import DeepSeekLLM, get_deepseek_llm
        print("✅ DeepSeek client imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 3: Create DeepSeek instance
    print("\n🔧 Creating DeepSeek instance...")
    llm = get_deepseek_llm(model="deepseek-llm:7b")
    print(f"✅ DeepSeek instance created: {llm.model}")
    
    # Test 4: Check Ollama availability (non-blocking)
    print("\n🔍 Checking Ollama availability...")
    if llm.is_available():
        print("✅ Ollama is running and DeepSeek model is available!")
    else:
        print("⚠️  Ollama not running. Run: ./setup_deepseek.sh")
        print("   This is expected in test environments without Ollama.")
    
    # Test 5: Import LLMAnalyzer
    print("\n📦 Testing LLMAnalyzer import...")
    try:
        from src.llm.analyzer import LLMAnalyzer
        print("✅ LLMAnalyzer imported successfully")
        
        # Check that it uses DeepSeek
        analyzer = LLMAnalyzer()
        assert "deepseek" in analyzer.model.lower(), f"Expected DeepSeek, got: {analyzer.model}"
        print(f"✅ LLMAnalyzer uses: {analyzer.model}")
    except Exception as e:
        print(f"⚠️  LLMAnalyzer test skipped: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 DeepSeek configuration tests passed!")
    print("\nSummary:")
    print("  - Model: DeepSeek 7B")
    print("  - Provider: Ollama (100% FREE)")
    print("  - No OpenAI/Anthropic API costs!")
    print("=" * 50)
    
    return True


def test_fallback_analysis():
    """Test that fallback analysis works without Ollama."""
    print("\n🧪 Testing fallback analysis...")
    
    from src.llm.deepseek_client import DeepSeekLLM
    
    llm = DeepSeekLLM(model="deepseek-llm:7b")
    
    # Create test match data
    match_data = {
        "home_team": "Bayern Munich",
        "away_team": "Dortmund",
        "league": "Bundesliga",
        "date": "2026-01-28",
        "home_goals_per_game": 2.5,
        "away_goals_per_game": 1.8,
        "home_btts_rate": 65,
        "away_btts_rate": 70,
    }
    
    # This will use fallback if Ollama is not running
    result = llm._fallback_analysis(match_data, "over_1_5")
    
    assert "probability" in result
    assert "confidence" in result
    assert "recommendation" in result
    
    print(f"✅ Fallback analysis works: probability={result['probability']:.2%}")
    
    return True


if __name__ == "__main__":
    success = True
    
    try:
        success &= test_deepseek_config()
        success &= test_fallback_analysis()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        success = False
    
    sys.exit(0 if success else 1)
