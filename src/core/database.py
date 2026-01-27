"""Database models and setup."""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from src.core.config import settings

Base = declarative_base()


class Match(Base):
    """Match data model."""
    
    __tablename__ = "matches"
    
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    league = Column(String, nullable=False)
    match_date = Column(DateTime, nullable=False)
    
    # Odds
    over_1_5_odds = Column(Float)
    btts_odds = Column(Float)
    home_win_odds = Column(Float)
    draw_odds = Column(Float)
    away_win_odds = Column(Float)
    
    # Features
    features = Column(JSON)
    
    # Results
    home_score = Column(Integer)
    away_score = Column(Integer)
    is_finished = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="match")


class Prediction(Base):
    """Prediction model."""
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    
    # Probabilities
    over_1_5_probability = Column(Float)
    btts_probability = Column(Float)
    
    # Model outputs
    statistical_probability = Column(Float)
    llm_probability = Column(Float)
    ensemble_probability = Column(Float)
    
    # Analysis
    llm_reasoning = Column(Text)
    key_factors = Column(JSON)
    confidence_score = Column(Float)
    
    # Value detection
    is_value_bet = Column(Boolean, default=False)
    expected_value = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    match = relationship("Match", back_populates="predictions")
    tips = relationship("Tip", back_populates="prediction")


class Tip(Base):
    """Betting tip model."""
    
    __tablename__ = "tips"
    
    id = Column(Integer, primary_key=True, index=True)
    tip_date = Column(DateTime, nullable=False)
    
    # Accumulator details
    is_accumulator = Column(Boolean, default=False)
    num_selections = Column(Integer, default=1)
    total_odds = Column(Float, nullable=False)
    
    # Bet details
    market_type = Column(String, nullable=False)  # over_1_5, btts, etc.
    stake_percentage = Column(Float)
    
    # Results
    is_sent = Column(Boolean, default=False)
    is_won = Column(Boolean)
    actual_return = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    prediction = relationship("Prediction", back_populates="tips")


class Bankroll(Base):
    """Bankroll tracking model."""
    
    __tablename__ = "bankroll"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False, unique=True)
    balance = Column(Float, nullable=False)
    profit_loss = Column(Float, default=0.0)
    num_bets = Column(Integer, default=0)
    num_wins = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)


# Database engine and session
engine = create_engine(
    settings.database.url,
    echo=settings.database.echo,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
