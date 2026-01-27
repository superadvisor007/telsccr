"""
📈 CLV Tracker - Closing Line Value Analysis
=============================================

Track Closing Line Value (CLV) - the ultimate indicator of betting skill.

Key insight from professional betting:
"Bettors who consistently beat the closing line are profitable 95%+ of the time."

Features:
- CLV calculation vs opening and closing odds
- Long-term CLV quality assessment
- Market efficiency analysis
- Skill vs luck attribution

Based on patterns from professional betting operations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLVQuality(Enum):
    """CLV quality tiers based on professional benchmarks"""
    ELITE = "Elite (top 1% of bettors)"
    PROFESSIONAL = "Professional (top 5%)"
    COMPETENT = "Competent (top 25%)"
    AMATEUR = "Amateur (average)"
    NEGATIVE = "Negative CLV (losing)"


@dataclass
class CLVEntry:
    """Single CLV measurement"""
    prediction_id: str
    match_date: str
    home_team: str
    away_team: str
    market: str
    
    # Odds chain
    predicted_probability: float
    odds_at_bet: float
    opening_odds: Optional[float] = None
    closing_odds: Optional[float] = None
    
    # CLV metrics
    clv_vs_close: Optional[float] = None  # Key metric
    clv_vs_open: Optional[float] = None
    implied_edge: Optional[float] = None
    
    # Result
    won: Optional[bool] = None
    pnl: Optional[float] = None


@dataclass
class CLVAnalysis:
    """Complete CLV analysis report"""
    timestamp: str
    total_bets: int
    bets_with_closing: int
    
    # Core CLV metrics
    avg_clv: float
    median_clv: float
    clv_stddev: float
    clv_positive_rate: float
    
    # Quality assessment
    clv_quality: CLVQuality
    quality_score: float  # 0-100
    
    # Performance correlation
    clv_pnl_correlation: float
    theoretical_roi: float  # Based on CLV alone
    actual_roi: float
    
    # Distribution
    clv_by_market: Dict[str, float]
    clv_by_confidence: Dict[str, float]
    clv_trend: List[float]  # Rolling 20-bet CLV
    
    # Insights
    best_market: str
    worst_market: str
    insights: List[str]
    
    def to_markdown(self) -> str:
        """Generate markdown report"""
        return f"""
# 📈 Closing Line Value (CLV) Analysis

**Generated:** {self.timestamp}
**Total Bets:** {self.total_bets}
**Bets with Closing Odds:** {self.bets_with_closing}

## 🎯 Core CLV Metrics

| Metric | Value |
|--------|-------|
| Average CLV | {self.avg_clv:+.2%} |
| Median CLV | {self.median_clv:+.2%} |
| CLV Std Dev | {self.clv_stddev:.2%} |
| CLV+ Rate | {self.clv_positive_rate:.1%} |

## ⭐ Quality Assessment

- **Quality Tier:** {self.clv_quality.value}
- **Quality Score:** {self.quality_score:.0f}/100
- **Assessment:** {"✅ Positive edge detected" if self.avg_clv > 0 else "❌ Negative edge"}

## 📊 Performance

| Metric | Value |
|--------|-------|
| CLV→PnL Correlation | {self.clv_pnl_correlation:.3f} |
| Theoretical ROI (CLV) | {self.theoretical_roi:+.2%} |
| Actual ROI | {self.actual_roi:+.2%} |

## 🏆 Market Performance

| Market | Avg CLV |
|--------|---------|
"""
        + '\n'.join(f"| {m} | {c:+.2%} |" for m, c in sorted(self.clv_by_market.items(), key=lambda x: -x[1]))
        + f"""

**Best Market:** {self.best_market}
**Worst Market:** {self.worst_market}

## 📈 CLV Trend (Rolling 20-bet)

```
{self._sparkline(self.clv_trend)}
```

## 💡 Insights

"""
        + '\n'.join(f"- {insight}" for insight in self.insights)
    
    @staticmethod
    def _sparkline(values: List[float], width: int = 50) -> str:
        """Generate ASCII sparkline"""
        if not values:
            return "No data"
        
        chars = ' ▁▂▃▄▅▆▇█'
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return chars[4] * min(len(values), width)
        
        step = len(values) / width if len(values) > width else 1
        result = []
        for i in range(min(len(values), width)):
            idx = int(i * step)
            normalized = (values[idx] - min_val) / (max_val - min_val)
            char_idx = int(normalized * (len(chars) - 1))
            result.append(chars[char_idx])
        
        return ''.join(result)


class CLVTracker:
    """
    Track and analyze Closing Line Value.
    
    CLV = (Your Odds - Closing Odds) / Closing Odds
    
    Professional benchmark:
    - Elite: +3% CLV average
    - Professional: +1.5% CLV
    - Competent: +0.5% CLV
    - Amateur: 0% CLV
    - Losing: <0% CLV
    """
    
    def __init__(self):
        self.entries: List[CLVEntry] = []
    
    def add_bet(
        self,
        prediction_id: str,
        match_date: str,
        home_team: str,
        away_team: str,
        market: str,
        predicted_probability: float,
        odds_at_bet: float,
        opening_odds: Optional[float] = None,
        closing_odds: Optional[float] = None,
        won: Optional[bool] = None,
        stake: float = 1.0
    ):
        """Add a bet to track"""
        entry = CLVEntry(
            prediction_id=prediction_id,
            match_date=match_date,
            home_team=home_team,
            away_team=away_team,
            market=market,
            predicted_probability=predicted_probability,
            odds_at_bet=odds_at_bet,
            opening_odds=opening_odds,
            closing_odds=closing_odds,
            won=won
        )
        
        # Calculate CLV
        if closing_odds:
            entry.clv_vs_close = (odds_at_bet - closing_odds) / closing_odds
        
        if opening_odds:
            entry.clv_vs_open = (odds_at_bet - opening_odds) / opening_odds
        
        # Implied edge
        implied_prob = 1 / odds_at_bet if odds_at_bet > 0 else 0
        entry.implied_edge = predicted_probability - implied_prob
        
        # PnL
        if won is not None:
            entry.pnl = (odds_at_bet - 1) * stake if won else -stake
        
        self.entries.append(entry)
    
    def add_closing_odds(
        self,
        prediction_id: str,
        closing_odds: float
    ):
        """Update closing odds for an existing bet"""
        for entry in self.entries:
            if entry.prediction_id == prediction_id:
                entry.closing_odds = closing_odds
                entry.clv_vs_close = (entry.odds_at_bet - closing_odds) / closing_odds
                break
    
    def add_result(self, prediction_id: str, won: bool, stake: float = 1.0):
        """Update result for an existing bet"""
        for entry in self.entries:
            if entry.prediction_id == prediction_id:
                entry.won = won
                entry.pnl = (entry.odds_at_bet - 1) * stake if won else -stake
                break
    
    def analyze(self) -> CLVAnalysis:
        """Generate full CLV analysis"""
        logger.info(f"Analyzing CLV for {len(self.entries)} entries...")
        
        # Filter entries with closing odds
        with_closing = [e for e in self.entries if e.clv_vs_close is not None]
        
        if not with_closing:
            logger.warning("No entries with closing odds")
            return self._empty_analysis()
        
        clvs = [e.clv_vs_close for e in with_closing]
        
        # Core metrics
        avg_clv = np.mean(clvs)
        median_clv = np.median(clvs)
        clv_stddev = np.std(clvs)
        clv_positive_rate = sum(1 for c in clvs if c > 0) / len(clvs)
        
        # Quality assessment
        quality, score = self._assess_quality(avg_clv, clv_positive_rate)
        
        # Performance correlation
        entries_with_result = [e for e in with_closing if e.pnl is not None]
        if entries_with_result:
            clv_arr = np.array([e.clv_vs_close for e in entries_with_result])
            pnl_arr = np.array([e.pnl for e in entries_with_result])
            correlation = np.corrcoef(clv_arr, pnl_arr)[0, 1] if len(clv_arr) > 1 else 0
            actual_roi = sum(pnl_arr) / len(pnl_arr)
        else:
            correlation = 0
            actual_roi = 0
        
        # Theoretical ROI from CLV
        theoretical_roi = self._theoretical_roi(avg_clv)
        
        # Market breakdown
        clv_by_market = {}
        for market in set(e.market for e in with_closing):
            market_clvs = [e.clv_vs_close for e in with_closing if e.market == market]
            clv_by_market[market] = np.mean(market_clvs)
        
        # Confidence breakdown
        clv_by_confidence = {}
        for bucket in ['50-60%', '60-70%', '70-80%', '80-100%']:
            bucket_entries = self._filter_by_confidence(with_closing, bucket)
            if bucket_entries:
                clv_by_confidence[bucket] = np.mean([e.clv_vs_close for e in bucket_entries])
        
        # Trend
        trend = self._calculate_trend(with_closing)
        
        # Best/worst
        best_market = max(clv_by_market.items(), key=lambda x: x[1])[0] if clv_by_market else "N/A"
        worst_market = min(clv_by_market.items(), key=lambda x: x[1])[0] if clv_by_market else "N/A"
        
        # Insights
        insights = self._generate_insights(
            avg_clv, clv_positive_rate, quality, clv_by_market, correlation
        )
        
        return CLVAnalysis(
            timestamp=datetime.now().isoformat(),
            total_bets=len(self.entries),
            bets_with_closing=len(with_closing),
            avg_clv=avg_clv,
            median_clv=median_clv,
            clv_stddev=clv_stddev,
            clv_positive_rate=clv_positive_rate,
            clv_quality=quality,
            quality_score=score,
            clv_pnl_correlation=correlation,
            theoretical_roi=theoretical_roi,
            actual_roi=actual_roi,
            clv_by_market=clv_by_market,
            clv_by_confidence=clv_by_confidence,
            clv_trend=trend,
            best_market=best_market,
            worst_market=worst_market,
            insights=insights
        )
    
    def _assess_quality(self, avg_clv: float, positive_rate: float) -> Tuple[CLVQuality, float]:
        """Assess CLV quality tier"""
        # Score formula: weighted combination
        score = (avg_clv * 1000) + (positive_rate * 50)
        score = max(0, min(100, score + 50))  # Normalize to 0-100
        
        if avg_clv >= 0.03 and positive_rate >= 0.60:
            return CLVQuality.ELITE, score
        elif avg_clv >= 0.015 and positive_rate >= 0.55:
            return CLVQuality.PROFESSIONAL, score
        elif avg_clv >= 0.005 and positive_rate >= 0.50:
            return CLVQuality.COMPETENT, score
        elif avg_clv >= 0:
            return CLVQuality.AMATEUR, score
        else:
            return CLVQuality.NEGATIVE, score
    
    def _theoretical_roi(self, avg_clv: float) -> float:
        """
        Calculate theoretical ROI from CLV.
        
        Research shows CLV roughly equals long-term ROI
        with some adjustment for variance.
        """
        # CLV to ROI is roughly 0.9x due to variance
        return avg_clv * 0.9
    
    def _filter_by_confidence(self, entries: List[CLVEntry], bucket: str) -> List[CLVEntry]:
        """Filter entries by confidence bucket"""
        ranges = {
            '50-60%': (0.50, 0.60),
            '60-70%': (0.60, 0.70),
            '70-80%': (0.70, 0.80),
            '80-100%': (0.80, 1.00)
        }
        lo, hi = ranges.get(bucket, (0, 1))
        return [e for e in entries if lo <= e.predicted_probability < hi]
    
    def _calculate_trend(self, entries: List[CLVEntry], window: int = 20) -> List[float]:
        """Calculate rolling CLV trend"""
        if len(entries) < window:
            return [e.clv_vs_close for e in entries]
        
        clvs = [e.clv_vs_close for e in entries]
        trend = []
        for i in range(window, len(clvs) + 1):
            trend.append(np.mean(clvs[i-window:i]))
        
        return trend
    
    def _generate_insights(
        self,
        avg_clv: float,
        positive_rate: float,
        quality: CLVQuality,
        by_market: Dict[str, float],
        correlation: float
    ) -> List[str]:
        """Generate actionable insights"""
        insights = []
        
        # Overall assessment
        if quality == CLVQuality.ELITE:
            insights.append("🏆 Elite CLV performance - strong long-term profitability expected")
        elif quality == CLVQuality.PROFESSIONAL:
            insights.append("📈 Professional-level CLV - above average betting skill")
        elif quality == CLVQuality.COMPETENT:
            insights.append("✅ Competent CLV - showing positive edge")
        elif quality == CLVQuality.AMATEUR:
            insights.append("⚠️ Amateur CLV - need to improve bet selection")
        else:
            insights.append("❌ Negative CLV - stop betting until strategy improves")
        
        # Positive rate
        if positive_rate > 0.55:
            insights.append(f"Strong {positive_rate:.0%} of bets beat closing line")
        elif positive_rate < 0.45:
            insights.append(f"Only {positive_rate:.0%} beat closing line - timing issue?")
        
        # Market insights
        if by_market:
            best = max(by_market.items(), key=lambda x: x[1])
            worst = min(by_market.items(), key=lambda x: x[1])
            
            if best[1] > 0.02:
                insights.append(f"Strong edge in {best[0]} market (+{best[1]:.1%} CLV)")
            if worst[1] < -0.01:
                insights.append(f"Consider avoiding {worst[0]} market ({worst[1]:+.1%} CLV)")
        
        # Correlation
        if correlation > 0.3:
            insights.append("Good CLV→PnL correlation - skill-based returns")
        elif correlation < 0:
            insights.append("Negative CLV→PnL correlation - possible variance/luck")
        
        return insights
    
    def _empty_analysis(self) -> CLVAnalysis:
        """Return empty analysis when no data"""
        return CLVAnalysis(
            timestamp=datetime.now().isoformat(),
            total_bets=len(self.entries),
            bets_with_closing=0,
            avg_clv=0,
            median_clv=0,
            clv_stddev=0,
            clv_positive_rate=0,
            clv_quality=CLVQuality.AMATEUR,
            quality_score=0,
            clv_pnl_correlation=0,
            theoretical_roi=0,
            actual_roi=0,
            clv_by_market={},
            clv_by_confidence={},
            clv_trend=[],
            best_market="N/A",
            worst_market="N/A",
            insights=["No closing odds data available"]
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    """Test CLV tracker with sample data"""
    tracker = CLVTracker()
    
    # Sample bets
    tracker.add_bet(
        prediction_id='1',
        match_date='2026-01-25',
        home_team='Bayern Munich',
        away_team='Wolfsburg',
        market='over_2_5',
        predicted_probability=0.78,
        odds_at_bet=1.65,
        opening_odds=1.70,
        closing_odds=1.55,
        won=True
    )
    
    tracker.add_bet(
        prediction_id='2',
        match_date='2026-01-25',
        home_team='Dortmund',
        away_team='Frankfurt',
        market='btts',
        predicted_probability=0.72,
        odds_at_bet=1.75,
        opening_odds=1.80,
        closing_odds=1.70,
        won=False
    )
    
    tracker.add_bet(
        prediction_id='3',
        match_date='2026-01-25',
        home_team='Leipzig',
        away_team='Mainz',
        market='over_1_5',
        predicted_probability=0.85,
        odds_at_bet=1.30,
        opening_odds=1.35,
        closing_odds=1.25,
        won=True
    )
    
    analysis = tracker.analyze()
    print(analysis.to_markdown())


if __name__ == '__main__':
    main()
