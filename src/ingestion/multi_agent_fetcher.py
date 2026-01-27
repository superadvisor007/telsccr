#!/usr/bin/env python3
"""
Multi-Agent Match Fetcher
Fetches matches from multiple APIs in parallel using async multi-agent pattern
Implements: Parallel execution, result aggregation, deduplication, fallback chains
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from loguru import logger

from src.core.production_api_client import ProductionAPIClient, production_api_call


class FetcherStrategy(Enum):
    """Match fetching strategies"""
    FASTEST = "fastest"  # Return first successful result
    MOST_COMPLETE = "most_complete"  # Wait for all, return most matches
    CONSENSUS = "consensus"  # Wait for all, deduplicate and merge


@dataclass
class MatchRecord:
    """Standardized match record"""
    home_team: str
    away_team: str
    league: str
    date: str
    time: str
    source: str
    match_hash: str = field(init=False)
    
    def __post_init__(self):
        """Generate unique hash for deduplication"""
        key = f"{self.home_team}_{self.away_team}_{self.date}_{self.league}".lower()
        self.match_hash = hashlib.md5(key.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'home_team': self.home_team,
            'away_team': self.away_team,
            'league': self.league,
            'date': self.date,
            'time': self.time,
            'source': self.source,
            'match_hash': self.match_hash
        }


class MatchFetcherAgent:
    """
    Individual agent responsible for one API source
    Each agent runs independently and reports back
    """
    
    def __init__(self, name: str, client: ProductionAPIClient):
        self.name = name
        self.client = client
        self.last_fetch_time: Optional[datetime] = None
        self.fetch_count: int = 0
        self.matches_found: int = 0
    
    @production_api_call(max_retries=2)
    async def fetch_upcoming_matches(self, date: str) -> List[MatchRecord]:
        """
        Fetch upcoming matches from this agent's API
        Returns standardized MatchRecord objects
        """
        raise NotImplementedError("Subclass must implement fetch_upcoming_matches")
    
    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            "name": self.name,
            "fetch_count": self.fetch_count,
            "matches_found": self.matches_found,
            "last_fetch": self.last_fetch_time.isoformat() if self.last_fetch_time else "Never",
            "client_metrics": self.client.get_metrics()
        }


class OpenLigaDBAgent(MatchFetcherAgent):
    """Agent for OpenLigaDB API (Bundesliga)"""
    
    async def fetch_upcoming_matches(self, date: str) -> List[MatchRecord]:
        """Fetch from OpenLigaDB"""
        self.fetch_count += 1
        self.last_fetch_time = datetime.now()
        
        logger.info(f"{self.name}: Fetching upcoming Bundesliga matches...")
        
        data = await self.client.get("getmatchdata/bl1")
        
        if not data:
            logger.warning(f"{self.name}: No data returned")
            return []
        
        matches = []
        now = datetime.now()
        
        for match in data:
            try:
                match_datetime_str = match.get('matchDateTime', '')
                if not match_datetime_str:
                    continue
                
                match_dt = datetime.fromisoformat(match_datetime_str.replace('Z', '+00:00'))
                
                # Only future matches
                if match_dt > now:
                    record = MatchRecord(
                        home_team=match['team1']['teamName'],
                        away_team=match['team2']['teamName'],
                        league='Bundesliga',
                        date=match_datetime_str[:10],
                        time=match_datetime_str[11:16],
                        source='OpenLigaDB'
                    )
                    matches.append(record)
                    
                    if len(matches) >= 10:  # Limit to 10
                        break
            
            except Exception as e:
                logger.debug(f"{self.name}: Error parsing match: {e}")
                continue
        
        self.matches_found += len(matches)
        logger.info(f"{self.name}: Found {len(matches)} upcoming matches")
        
        return matches


class TheSportsDBAgent(MatchFetcherAgent):
    """Agent for TheSportsDB API (Multiple leagues)"""
    
    LEAGUES = {
        'Premier League': '4328',
        'La Liga': '4335',
        'Serie A': '4332',
        'Ligue 1': '4334'
    }
    
    async def fetch_upcoming_matches(self, date: str) -> List[MatchRecord]:
        """Fetch from TheSportsDB across multiple leagues"""
        self.fetch_count += 1
        self.last_fetch_time = datetime.now()
        
        logger.info(f"{self.name}: Fetching matches from {len(self.LEAGUES)} leagues...")
        
        all_matches = []
        now = datetime.now()
        
        # Check next 7 days
        for days_ahead in range(1, 8):
            check_date = (now + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            
            for league_name, league_id in self.LEAGUES.items():
                try:
                    endpoint = f"eventsday.php?d={check_date}&l={league_id}"
                    data = await self.client.get(endpoint)
                    
                    if not data:
                        continue
                    
                    events = data.get('events') or []
                    
                    for event in events:
                        if event and event.get('strSport') == 'Soccer':
                            record = MatchRecord(
                                home_team=event['strHomeTeam'],
                                away_team=event['strAwayTeam'],
                                league=league_name,
                                date=check_date,
                                time=event.get('strTime', 'TBD'),
                                source='TheSportsDB'
                            )
                            all_matches.append(record)
                    
                    # Small delay between leagues
                    await asyncio.sleep(0.5)
                
                except Exception as e:
                    logger.debug(f"{self.name}: Error fetching {league_name}: {e}")
                    continue
            
            # Stop if we found enough matches
            if len(all_matches) >= 20:
                break
        
        self.matches_found += len(all_matches)
        logger.info(f"{self.name}: Found {len(all_matches)} upcoming matches")
        
        return all_matches


class FootballDataAgent(MatchFetcherAgent):
    """Agent for Football-Data.org API (requires API key)"""
    
    async def fetch_upcoming_matches(self, date: str) -> List[MatchRecord]:
        """Fetch from Football-Data.org"""
        self.fetch_count += 1
        self.last_fetch_time = datetime.now()
        
        if not self.client.api_key:
            logger.warning(f"{self.name}: No API key configured, skipping")
            return []
        
        logger.info(f"{self.name}: Fetching matches...")
        
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        week_later = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        data = await self.client.get(
            "matches",
            params={"dateFrom": tomorrow, "dateTo": week_later}
        )
        
        if not data:
            logger.warning(f"{self.name}: No data returned")
            return []
        
        matches = []
        for match in data.get('matches', []):
            try:
                record = MatchRecord(
                    home_team=match['homeTeam']['name'],
                    away_team=match['awayTeam']['name'],
                    league=match['competition']['name'],
                    date=match['utcDate'][:10],
                    time=match['utcDate'][11:16],
                    source='Football-Data.org'
                )
                matches.append(record)
            except Exception as e:
                logger.debug(f"{self.name}: Error parsing match: {e}")
                continue
        
        self.matches_found += len(matches)
        logger.info(f"{self.name}: Found {len(matches)} upcoming matches")
        
        return matches


class MultiAgentMatchFetcher:
    """
    Orchestrator for multi-agent match fetching
    Coordinates multiple agents to fetch matches in parallel
    """
    
    def __init__(self, strategy: FetcherStrategy = FetcherStrategy.CONSENSUS):
        self.strategy = strategy
        self.agents: List[MatchFetcherAgent] = []
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info(f"MultiAgentMatchFetcher initialized with {len(self.agents)} agents")
        logger.info(f"Strategy: {strategy.value}")
    
    def _initialize_agents(self):
        """Initialize all available agents"""
        
        # OpenLigaDB Agent (always available - free, no key)
        openligadb_client = ProductionAPIClient(
            name="OpenLigaDB",
            base_url="https://api.openligadb.de",
            rate_limit_requests=100,
            rate_limit_window=60
        )
        self.agents.append(OpenLigaDBAgent("OpenLigaDB-Agent", openligadb_client))
        
        # TheSportsDB Agent (always available - free, no key)
        thesportsdb_client = ProductionAPIClient(
            name="TheSportsDB",
            base_url="https://www.thesportsdb.com/api/v1/json/3",
            rate_limit_requests=100,
            rate_limit_window=60
        )
        self.agents.append(TheSportsDBAgent("TheSportsDB-Agent", thesportsdb_client))
        
        # Football-Data.org Agent (optional - requires key)
        # Uncomment if you have API key:
        # football_data_client = ProductionAPIClient(
        #     name="FootballData",
        #     base_url="https://api.football-data.org/v4",
        #     api_key="YOUR_API_KEY_HERE",
        #     rate_limit_requests=10,
        #     rate_limit_window=60
        # )
        # self.agents.append(FootballDataAgent("FootballData-Agent", football_data_client))
    
    async def fetch_matches(self, date: Optional[str] = None) -> List[Dict]:
        """
        Fetch matches from all agents in parallel
        
        Args:
            date: Target date (YYYY-MM-DD) or None for tomorrow
        
        Returns:
            List of deduplicated match dictionaries
        """
        if not date:
            date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info(f"🔍 Fetching matches for {date} using {self.strategy.value} strategy...")
        
        # Create tasks for all agents
        tasks = [
            agent.fetch_upcoming_matches(date)
            for agent in self.agents
        ]
        
        # Execute in parallel
        start_time = datetime.now()
        
        if self.strategy == FetcherStrategy.FASTEST:
            # Return first successful result
            results = await self._fetch_fastest(tasks)
        elif self.strategy == FetcherStrategy.MOST_COMPLETE:
            # Wait for all, return most complete
            results = await self._fetch_most_complete(tasks)
        else:  # CONSENSUS
            # Wait for all, deduplicate and merge
            results = await self._fetch_consensus(tasks)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Fetched {len(results)} matches in {elapsed:.1f}s")
        
        return [match.to_dict() for match in results]
    
    async def _fetch_fastest(self, tasks: List) -> List[MatchRecord]:
        """Strategy: Return first successful result"""
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
        
        # Return first result
        for task in done:
            result = task.result()
            if result:
                return result
        
        return []
    
    async def _fetch_most_complete(self, tasks: List) -> List[MatchRecord]:
        """Strategy: Wait for all, return most complete dataset"""
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Find result with most matches
        best_result = []
        for result in results:
            if isinstance(result, list) and len(result) > len(best_result):
                best_result = result
        
        return best_result
    
    async def _fetch_consensus(self, tasks: List) -> List[MatchRecord]:
        """Strategy: Wait for all, deduplicate and merge"""
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Deduplicate using match hash
        seen_hashes: Set[str] = set()
        merged_matches: List[MatchRecord] = []
        
        for result in results:
            if not isinstance(result, list):
                continue
            
            for match in result:
                if match.match_hash not in seen_hashes:
                    seen_hashes.add(match.match_hash)
                    merged_matches.append(match)
        
        # Sort by date, then league
        merged_matches.sort(key=lambda m: (m.date, m.league, m.home_team))
        
        logger.info(f"Deduplicated from {sum(len(r) if isinstance(r, list) else 0 for r in results)} to {len(merged_matches)} unique matches")
        
        return merged_matches
    
    def get_agent_status(self) -> List[Dict]:
        """Get status of all agents"""
        return [agent.get_status() for agent in self.agents]
    
    async def close(self):
        """Close all agent clients"""
        for agent in self.agents:
            await agent.client.close()
        logger.info("All agent clients closed")


# Example usage and testing
async def main():
    """Test the multi-agent fetcher"""
    print("\n" + "="*80)
    print("🤖 MULTI-AGENT MATCH FETCHER TEST")
    print("="*80 + "\n")
    
    # Initialize fetcher
    fetcher = MultiAgentMatchFetcher(strategy=FetcherStrategy.CONSENSUS)
    
    # Fetch matches
    matches = await fetcher.fetch_matches()
    
    # Display results
    print(f"\n✅ Found {len(matches)} unique matches:\n")
    
    for i, match in enumerate(matches[:10], 1):  # Show first 10
        print(f"{i}. {match['home_team']} vs {match['away_team']}")
        print(f"   📅 {match['date']} {match['time']} | 🏆 {match['league']}")
        print(f"   📡 Source: {match['source']}\n")
    
    # Show agent status
    print("\n" + "="*80)
    print("📊 AGENT STATUS")
    print("="*80 + "\n")
    
    for status in fetcher.get_agent_status():
        print(f"🤖 {status['name']}")
        print(f"   Fetches: {status['fetch_count']}")
        print(f"   Matches Found: {status['matches_found']}")
        print(f"   Last Fetch: {status['last_fetch']}")
        print(f"   Client: {status['client_metrics']['success_rate']} success rate")
        print()
    
    # Cleanup
    await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
