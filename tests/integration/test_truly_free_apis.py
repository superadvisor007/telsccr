#!/usr/bin/env python3
"""
Test truly free APIs - NO PAID SERVICES!

Tests:
1. TheSportsDB (NO API KEY NEEDED)
2. OpenLigaDB (NO API KEY NEEDED)  
3. Football-Data.org (simple email signup)
"""
import asyncio
import os
from dotenv import load_dotenv

from src.ingestion.truly_free_apis import (
    FootballDataOrgClient,
    TheSportsDBClient,
    OpenLigaDBClient,
    TrulyFreeQuotaManager
)

load_dotenv()


async def test_thesportsdb():
    """Test TheSportsDB - NO KEY NEEDED!"""
    print("\n" + "="*70)
    print("1️⃣  TESTING TheSportsDB (NO API KEY REQUIRED)")
    print("="*70)
    
    try:
        client = TheSportsDBClient()  # No key needed!
        
        print("\n📋 Fetching next Premier League matches...")
        matches = await client.get_next_matches("English Premier League")
        
        if matches:
            print(f"✓ VALID: Found {len(matches)} upcoming matches")
            
            # Show first match
            if matches:
                m = matches[0]
                print(f"\n   Example: {m.get('strHomeTeam', 'N/A')} vs {m.get('strAwayTeam', 'N/A')}")
                print(f"   Date: {m.get('dateEvent', 'N/A')}")
            
            return True
        else:
            print("⚠️  No matches found (might be off-season)")
            return True  # Still valid, just no data
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


async def test_openligadb():
    """Test OpenLigaDB - NO KEY NEEDED!"""
    print("\n" + "="*70)
    print("2️⃣  TESTING OpenLigaDB (NO API KEY REQUIRED)")
    print("="*70)
    
    try:
        client = OpenLigaDBClient()  # No key needed!
        
        print("\n📋 Fetching current Bundesliga matchday...")
        matches = await client.get_current_matchday("bl1")
        
        if matches:
            print(f"✓ VALID: Found {len(matches)} Bundesliga matches")
            
            # Show first match
            if matches:
                m = matches[0]
                team1 = m.get('team1', {}).get('teamName', 'N/A')
                team2 = m.get('team2', {}).get('teamName', 'N/A')
                print(f"\n   Example: {team1} vs {team2}")
                print(f"   Date: {m.get('matchDateTime', 'N/A')}")
            
            return True
        else:
            print("⚠️  No matches found (might be off-season)")
            return True  # Still valid
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


async def test_football_data_org():
    """Test Football-Data.org (requires free key)."""
    print("\n" + "="*70)
    print("3️⃣  TESTING Football-Data.org (requires free key)")
    print("="*70)
    
    api_key = os.getenv("FOOTBALL_DATA_ORG_KEY")
    
    if not api_key:
        print("\n⏳ API Key not configured")
        print("\n📝 To get your FREE key:")
        print("   1. Visit: https://www.football-data.org/client/register")
        print("   2. Enter your email (no credit card!)")
        print("   3. Check inbox for API key")
        print("   4. Add to .env: FOOTBALL_DATA_ORG_KEY=your_key")
        print("\n💰 Cost: $0 forever (10 requests/minute)")
        return None  # Not configured, skip
    
    try:
        client = FootballDataOrgClient(api_key)
        
        print("\n📋 Fetching today's matches...")
        matches = await client.get_todays_matches()
        
        if matches:
            print(f"✓ VALID: Found {len(matches)} matches today")
            
            # Show first match
            if matches:
                m = matches[0]
                home = m.get('homeTeam', {}).get('name', 'N/A')
                away = m.get('awayTeam', {}).get('name', 'N/A')
                print(f"\n   Example: {home} vs {away}")
                print(f"   Competition: {m.get('competition', {}).get('name', 'N/A')}")
            
            return True
        else:
            print("⚠️  No matches today")
            return True  # Still valid
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            print("✗ INVALID KEY: Check your API key")
        else:
            print(f"✗ FAILED: HTTP {e.response.status_code}")
        return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


async def test_quota_manager():
    """Test multi-source quota manager."""
    print("\n" + "="*70)
    print("4️⃣  TESTING TrulyFreeQuotaManager (Multi-Source)")
    print("="*70)
    
    try:
        football_data_key = os.getenv("FOOTBALL_DATA_ORG_KEY")
        
        manager = TrulyFreeQuotaManager(
            football_data_key=football_data_key
        )
        
        print(f"\n📊 Initialized with {len(manager.apis)} data sources")
        for api_name in manager.apis.keys():
            print(f"   • {api_name}")
        
        print("\n📋 Fetching today's matches from best source...")
        matches = await manager.get_todays_matches()
        
        if matches:
            print(f"✓ SUCCESS: Found {len(matches)} matches")
            return True
        else:
            print("⚠️  No matches found from any source")
            return True  # Still valid
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "🎯"*35)
    print("  TESTING 100% FOREVER FREE SOCCER APIs")
    print("  No hidden costs, no credit cards, no bullshit!")
    print("🎯"*35)
    
    results = []
    
    # Test 1: TheSportsDB (always works, no key)
    results.append(("TheSportsDB", await test_thesportsdb()))
    
    # Test 2: OpenLigaDB (always works, no key)
    results.append(("OpenLigaDB", await test_openligadb()))
    
    # Test 3: Football-Data.org (optional, needs free key)
    result = await test_football_data_org()
    if result is not None:  # Only add if configured
        results.append(("Football-Data.org", result))
    
    # Test 4: Multi-source manager
    results.append(("QuotaManager", await test_quota_manager()))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<10} {name}")
    
    print(f"\n🎯 Result: {passed}/{total} tests passed")
    
    if passed >= 2:  # At least 2 APIs working
        print("\n✅ READY TO USE!")
        print("💰 Cost: $0.00 FOREVER")
        print("\n🚀 Next steps:")
        print("   1. APIs are working without configuration")
        print("   2. Optional: Get Football-Data.org key for more data")
        print("   3. Run: python src/pipeline_free.py")
    else:
        print("\n⚠️  WARNING: Not enough APIs working")
        print("Please check your network connection.")
    
    return passed >= 2


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
