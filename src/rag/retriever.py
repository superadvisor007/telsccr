"""RAG system for retrieving similar past betting mistakes."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.core.database import DatabaseManager, Match, Prediction, Tip


class BettingMemoryRAG:
    """
    Retrieval-Augmented Generation system for betting analysis.
    
    Stores failed tips and post-mortems in a vector database,
    then retrieves similar situations during new match analysis.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        chroma_path: str = "data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.db = db_manager
        self.chroma_path = Path(chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Initialize embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="betting_mistakes",
            embedding_function=self.embedding_fn,
            metadata={"description": "Failed betting tips with analysis"},
        )
        
        logger.info(f"RAG system initialized with {self.collection.count()} memories")
    
    async def index_failed_tips(
        self,
        start_date: Optional[datetime] = None,
        reindex_all: bool = False,
    ) -> int:
        """
        Index failed betting tips into vector database.
        
        Returns:
            Number of tips indexed
        """
        if reindex_all:
            logger.info("Clearing existing memories for re-indexing...")
            self.chroma_client.delete_collection("betting_mistakes")
            self.collection = self.chroma_client.create_collection(
                name="betting_mistakes",
                embedding_function=self.embedding_fn,
            )
        
        # Fetch failed tips
        session = self.db.get_session()
        query = session.query(Tip, Match, Prediction).join(
            Match, Tip.match_id == Match.id
        ).join(
            Prediction, Tip.prediction_id == Prediction.id
        ).filter(
            Tip.result == "lost"
        )
        
        if start_date and not reindex_all:
            query = query.filter(Match.date >= start_date)
        
        failed_tips = query.all()
        session.close()
        
        if not failed_tips:
            logger.info("No failed tips to index")
            return 0
        
        # Prepare documents for indexing
        documents = []
        metadatas = []
        ids = []
        
        for tip, match, prediction in failed_tips:
            # Create searchable document
            doc_text = self._create_memory_document(tip, match, prediction)
            
            # Metadata for filtering
            metadata = {
                "match_id": match.id,
                "league": match.league,
                "market": tip.market,
                "odds": tip.odds,
                "probability": tip.probability,
                "date": match.date.isoformat(),
                "post_mortem": tip.post_mortem or "",
            }
            
            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(f"tip_{tip.id}")
        
        # Add to ChromaDB in batches
        batch_size = 100
        indexed_count = 0
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            indexed_count += len(batch_docs)
        
        logger.info(f"Indexed {indexed_count} failed tips into RAG system")
        return indexed_count
    
    def _create_memory_document(
        self,
        tip: Tip,
        match: Match,
        prediction: Prediction,
    ) -> str:
        """Create searchable text document from failed tip."""
        return f"""Failed Betting Tip Memory

Match: {match.home_team} vs {match.away_team}
League: {match.league}
Date: {match.date.strftime('%Y-%m-%d')}
Market: {tip.market} @ {tip.odds}
Predicted Probability: {tip.probability:.1%}
Result: LOST (Score: {match.home_score}-{match.away_score})

Pre-Match Analysis:
{prediction.reasoning}

Key Factors Considered:
{', '.join(prediction.key_factors or [])}

What Went Wrong:
{tip.post_mortem or 'Analysis unavailable'}

Team Stats:
Home - Goals/Game: {match.home_goals_per_game:.2f}, Form: {match.home_form_ppg:.2f}
Away - Goals/Game: {match.away_goals_per_game:.2f}, Form: {match.away_form_ppg:.2f}

Weather: {match.weather_description or 'N/A'} ({match.weather_temp or 15}°C)
"""
    
    def retrieve_similar_mistakes(
        self,
        query_match: Dict,
        n_results: int = 3,
        league_filter: Optional[str] = None,
        market_filter: Optional[str] = None,
        days_back: int = 365,
    ) -> List[Dict]:
        """
        Retrieve similar past betting mistakes for a new match.
        
        Args:
            query_match: Dict with match details (teams, league, stats, etc.)
            n_results: Number of similar cases to retrieve
            league_filter: Only retrieve mistakes from this league
            market_filter: Only retrieve mistakes for this market
            days_back: Only consider mistakes from last N days
        
        Returns:
            List of similar mistake dicts with metadata and analysis
        """
        # Create query text from match details
        query_text = self._format_query(query_match)
        
        # Prepare filters
        where_filter = {}
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        where_filter["date"] = {"$gte": cutoff_date}
        
        if league_filter:
            where_filter["league"] = league_filter
        if market_filter:
            where_filter["market"] = market_filter
        
        # Query ChromaDB
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter if where_filter else None,
            )
            
            if not results['documents'] or not results['documents'][0]:
                logger.debug("No similar mistakes found")
                return []
            
            # Format results
            similar_mistakes = []
            for i, doc in enumerate(results['documents'][0]):
                similar_mistakes.append({
                    'document': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                })
            
            logger.debug(f"Retrieved {len(similar_mistakes)} similar mistakes")
            return similar_mistakes
        
        except Exception as e:
            logger.error(f"Error retrieving similar mistakes: {e}")
            return []
    
    def _format_query(self, match: Dict) -> str:
        """Format match details into query text."""
        return f"""Upcoming match analysis:

Teams: {match.get('home_team')} vs {match.get('away_team')}
League: {match.get('league')}
Market: {match.get('market', 'Over 1.5 Goals')}

Home Team:
- Goals/Game: {match.get('home_goals_per_game', 0):.2f}
- Form: {match.get('home_form_ppg', 0):.2f} PPG
- Recent: {match.get('home_recent_form', 'N/A')}

Away Team:
- Goals/Game: {match.get('away_goals_per_game', 0):.2f}
- Form: {match.get('away_form_ppg', 0):.2f} PPG
- Recent: {match.get('away_recent_form', 'N/A')}

Context: {match.get('context', '')}
"""
    
    def generate_rag_enhanced_prompt(
        self,
        base_prompt: str,
        query_match: Dict,
        market: str,
    ) -> str:
        """
        Enhance LLM prompt with retrieved similar mistakes.
        
        Args:
            base_prompt: Original match analysis prompt
            query_match: Match details for retrieval
            market: Target betting market
        
        Returns:
            Enhanced prompt with RAG context
        """
        # Retrieve similar mistakes
        similar_mistakes = self.retrieve_similar_mistakes(
            query_match=query_match,
            n_results=2,
            market_filter=market,
        )
        
        if not similar_mistakes:
            return base_prompt  # No enhancement if no similar cases
        
        # Build RAG context section
        rag_context = "\n\n### 🧠 LEARN FROM PAST MISTAKES\n\n"
        rag_context += "Recall these similar situations where we made errors:\n\n"
        
        for i, mistake in enumerate(similar_mistakes, 1):
            rag_context += f"**Similar Case {i}**:\n"
            rag_context += f"- Match: {mistake['metadata'].get('match_id', 'N/A')}\n"
            rag_context += f"- What happened: {mistake['metadata'].get('post_mortem', 'N/A')[:200]}...\n"
            rag_context += f"- Distance score: {mistake.get('distance', 0):.3f}\n\n"
        
        rag_context += "**CRITICAL**: Ensure you don't repeat these mistakes in your current analysis!\n"
        
        # Insert RAG context before the analysis instructions
        enhanced_prompt = base_prompt.replace(
            "Provide probability estimate",
            f"{rag_context}\nNow, with these lessons in mind, provide probability estimate"
        )
        
        return enhanced_prompt
    
    def get_statistics(self) -> Dict:
        """Get RAG system statistics."""
        return {
            'total_memories': self.collection.count(),
            'chroma_path': str(self.chroma_path),
            'embedding_model': 'all-MiniLM-L6-v2',
        }
