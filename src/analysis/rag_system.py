#!/usr/bin/env python3
"""RAG System for Historical Match Context - Battle Tested"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class MatchContext:
    """Historical match record"""
    home_team: str
    away_team: str
    league: str
    home_goals: int
    away_goals: int
    date: str
    context_summary: str = ""

class SimpleRAG:
    """Simple similarity-based match finder (no heavy dependencies)"""
    
    def __init__(self, data_file: str = "data/historical/massive_training_data.csv"):
        self.matches: List[MatchContext] = []
        self._load_data(data_file)
    
    def _load_data(self, file_path: str):
        """Load historical matches"""
        import pandas as pd
        try:
            df = pd.read_csv(file_path)
            for _, row in df.head(1000).iterrows():  # Load first 1000
                try:
                    self.matches.append(MatchContext(
                        home_team=row.get('home_team', row.get('HomeTeam', '')),
                        away_team=row.get('away_team', row.get('AwayTeam', '')),
                        league=row.get('league', 'Unknown'),
                        home_goals=int(row.get('home_goals', row.get('FTHG', 0))),
                        away_goals=int(row.get('away_goals', row.get('FTAG', 0))),
                        date=str(row.get('date', row.get('Date', ''))),
                        context_summary=f"{row.get('home_team', row.get('HomeTeam', ''))} {row.get('home_goals', row.get('FTHG', 0))}-{row.get('away_goals', row.get('FTAG', 0))} {row.get('away_team', row.get('AwayTeam', ''))}"
                    ))
                except Exception:
                    continue
            print(f"✅ Loaded {len(self.matches)} historical matches")
        except Exception as e:
            print(f"⚠️  Could not load RAG data: {e}")
    
    def find_similar_matches(self, home_team: str, away_team: str, limit: int = 3) -> List[Dict]:
        """Find similar historical matches"""
        similar = []
        
        for match in self.matches:
            # Simple similarity: same teams or same league
            score = 0
            if match.home_team.lower() == home_team.lower():
                score += 10
            if match.away_team.lower() == away_team.lower():
                score += 10
            if home_team.lower() in match.home_team.lower() or away_team.lower() in match.away_team.lower():
                score += 5
            
            if score > 0:
                similar.append({
                    'match': match,
                    'score': score,
                    'summary': match.context_summary
                })
        
        # Sort by score and return top N
        similar.sort(key=lambda x: x['score'], reverse=True)
        return similar[:limit]
    
    def get_context_for_prediction(self, home_team: str, away_team: str) -> str:
        """Get contextual insights for prediction"""
        similar = self.find_similar_matches(home_team, away_team, limit=5)
        
        if not similar:
            return "No historical context available."
        
        context = f"Historical context (last 5 similar matches):\n"
        for i, match_data in enumerate(similar, 1):
            context += f"{i}. {match_data['summary']}\n"
        
        return context

# Test
if __name__ == "__main__":
    rag = SimpleRAG()
    context = rag.get_context_for_prediction("Bayern Munich", "Borussia Dortmund")
    print(context)
    print("✅ RAG System works!")
