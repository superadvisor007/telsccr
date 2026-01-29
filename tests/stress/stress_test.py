"""
Comprehensive Stress Test Suite
Tests ML system on 10,000+ matches with visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os
from datetime import datetime
import json

# Import walk-forward backtester
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.testing.walk_forward_backtest import WalkForwardBacktester, SimpleGradientBoostModel

class StressTestSuite:
    """
    Comprehensive testing on 10K+ matches:
    - Walk-forward backtest
    - Benchmark comparisons
    - Visualizations
    - Performance reports
    """
    
    def __init__(self, data_path: str = 'data/historical/massive_training_data.csv'):
        self.data_path = data_path
        self.results = {}
        self.output_dir = 'data/stress_tests'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_full_stress_test(self) -> Dict:
        """Execute complete stress test battery"""
        print("\n" + "="*80)
        print("🏋️  10,000+ MATCH STRESS TEST SUITE")
        print("="*80)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Load data
        print("\n📂 Loading data...")
        data = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(data)} matches")
        
        self.results['data_info'] = {
            'total_matches': len(data),
            'date_range': f"{data['date'].min()} to {data['date'].max()}",
            'leagues': data['league'].nunique(),
            'teams': len(set(data['home_team'].unique()) | set(data['away_team'].unique()))
        }
        
        # Test 1: Walk-Forward Backtest
        print("\n" + "-"*80)
        print("TEST 1: Walk-Forward Backtest (Primary Test)")
        print("-"*80)
        wf_results = self._run_walk_forward_test(data)
        self.results['walk_forward'] = wf_results
        
        # Test 2: Benchmark against Random
        print("\n" + "-"*80)
        print("TEST 2: Benchmark vs Random Betting")
        print("-"*80)
        random_results = self._benchmark_random_betting(data)
        self.results['random_benchmark'] = random_results
        
        # Test 3: Benchmark against Favorites
        print("\n" + "-"*80)
        print("TEST 3: Benchmark vs Always Bet Favorites")
        print("-"*80)
        favorite_results = self._benchmark_favorite_betting(data)
        self.results['favorite_benchmark'] = favorite_results
        
        # Test 4: Market-Specific Performance
        print("\n" + "-"*80)
        print("TEST 4: Market-Specific Analysis")
        print("-"*80)
        market_results = self._analyze_by_market(wf_results)
        self.results['market_analysis'] = market_results
        
        # Test 5: League-Specific Performance
        print("\n" + "-"*80)
        print("TEST 5: League-Specific Analysis")
        print("-"*80)
        league_results = self._analyze_by_league(data, wf_results)
        self.results['league_analysis'] = league_results
        
        # Generate visualizations
        print("\n" + "-"*80)
        print("📊 Generating Visualizations...")
        print("-"*80)
        self._generate_visualizations(wf_results, timestamp)
        
        # Generate report
        print("\n" + "-"*80)
        print("📄 Generating Report...")
        print("-"*80)
        report_path = self._generate_report(timestamp)
        
        print("\n" + "="*80)
        print("✅ STRESS TEST COMPLETE")
        print("="*80)
        print(f"📄 Report: {report_path}")
        print(f"📊 Charts: {self.output_dir}/charts_{timestamp}/")
        print("="*80)
        
        return self.results
    
    def _run_walk_forward_test(self, data: pd.DataFrame) -> Dict:
        """Run walk-forward backtest"""
        backtester = WalkForwardBacktester(
            train_window=500,
            test_window=50,
            step_size=50,
            min_edge=0.08,
            kelly_fraction=0.25
        )
        
        results = backtester.run_backtest(
            data=data,
            model_class=SimpleGradientBoostModel,
            markets=['over_1_5', 'over_2_5', 'btts'],
            initial_bankroll=1000.0
        )
        
        backtester.print_results(results)
        
        return results
    
    def _benchmark_random_betting(self, data: pd.DataFrame) -> Dict:
        """Simulate random betting strategy"""
        print("\n🎲 Simulating random betting...")
        
        # Sample 20% of matches randomly
        sample_size = int(len(data) * 0.20)
        sample = data.sample(sample_size, random_state=42)
        
        bankroll = 1000.0
        total_bets = 0
        wins = 0
        
        for _, match in sample.iterrows():
            # Random market
            market = np.random.choice(['over_1_5', 'over_2_5', 'btts'])
            
            # Fixed stake 2%
            stake = bankroll * 0.02
            
            # Use realistic odds based on market probability
            # Over 1.5 typically 75% → odds ~1.33
            # Over 2.5 typically 50% → odds ~2.00
            # BTTS typically 50% → odds ~2.00
            if market == 'over_1_5':
                odds = np.random.uniform(1.20, 1.50)
            elif market == 'over_2_5':
                odds = np.random.uniform(1.70, 2.30)
            else:  # btts
                odds = np.random.uniform(1.80, 2.20)
            
            # Evaluate
            total_goals = match['home_goals'] + match['away_goals']
            if market == 'over_1_5':
                won = total_goals > 1.5
            elif market == 'over_2_5':
                won = total_goals > 2.5
            else:  # btts
                won = (match['home_goals'] > 0 and match['away_goals'] > 0)
            
            if won:
                profit = stake * odds - stake
                wins += 1
            else:
                profit = -stake
            
            bankroll += profit
            total_bets += 1
            
            # Bankroll protection
            if bankroll < 100:
                break
        
        roi = ((bankroll - 1000) / 1000) * 100
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        results = {
            'total_bets': total_bets,
            'wins': wins,
            'win_rate': win_rate,
            'final_bankroll': bankroll,
            'roi': roi
        }
        
        print(f"\n📊 Random Betting Results:")
        print(f"   Win Rate: {win_rate*100:.1f}%")
        print(f"   ROI:      {roi:+.2f}%")
        print(f"   Bankroll: ${bankroll:.2f}")
        
        return results
    
    def _benchmark_favorite_betting(self, data: pd.DataFrame) -> Dict:
        """Simulate always betting on favorites (home win or over 1.5)"""
        print("\n⭐ Simulating favorite betting...")
        
        # Sample 20% of matches
        sample_size = int(len(data) * 0.20)
        sample = data.sample(sample_size, random_state=42)
        
        bankroll = 1000.0
        total_bets = 0
        wins = 0
        
        for _, match in sample.iterrows():
            # Always bet Over 1.5 (most likely in major leagues)
            stake = bankroll * 0.02
            odds = 1.20  # Typical odds for Over 1.5
            
            total_goals = match['home_goals'] + match['away_goals']
            won = total_goals > 1.5
            
            if won:
                profit = stake * odds - stake
                wins += 1
            else:
                profit = -stake
            
            bankroll += profit
            total_bets += 1
        
        roi = ((bankroll - 1000) / 1000) * 100
        win_rate = wins / total_bets
        
        results = {
            'total_bets': total_bets,
            'wins': wins,
            'win_rate': win_rate,
            'final_bankroll': bankroll,
            'roi': roi
        }
        
        print(f"\n📊 Favorite Betting Results:")
        print(f"   Win Rate: {win_rate*100:.1f}%")
        print(f"   ROI:      {roi:+.2f}%")
        print(f"   Bankroll: ${bankroll:.2f}")
        
        return results
    
    def _analyze_by_market(self, wf_results: Dict) -> Dict:
        """Analyze performance by market"""
        if 'raw_bets' not in wf_results:
            return {}
        
        df = pd.DataFrame(wf_results['raw_bets'])
        
        market_stats = df.groupby('market').agg({
            'won': ['count', 'sum', 'mean'],
            'profit': 'sum',
            'stake': 'sum',
            'edge': 'mean'
        }).round(3)
        
        results = {}
        for market in df['market'].unique():
            market_bets = df[df['market'] == market]
            results[market] = {
                'total_bets': len(market_bets),
                'wins': market_bets['won'].sum(),
                'win_rate': market_bets['won'].mean(),
                'total_profit': market_bets['profit'].sum(),
                'roi': (market_bets['profit'].sum() / market_bets['stake'].sum()) * 100 if market_bets['stake'].sum() > 0 else 0
            }
        
        print("\n📊 Market Performance:")
        for market, stats in results.items():
            print(f"\n   {market.upper()}:")
            print(f"      Bets:     {stats['total_bets']}")
            print(f"      Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"      ROI:      {stats['roi']:+.2f}%")
        
        return results
    
    def _analyze_by_league(self, data: pd.DataFrame, wf_results: Dict) -> Dict:
        """Analyze performance by league"""
        # This would require league info in bets, simplified version
        leagues = data['league'].unique()
        
        results = {}
        for league in leagues:
            league_data = data[data['league'] == league]
            results[league] = {
                'total_matches': len(league_data),
                'avg_goals': league_data['total_goals'].mean(),
                'over_2_5_rate': (league_data['total_goals'] > 2.5).mean(),
                'btts_rate': ((league_data['home_goals'] > 0) & (league_data['away_goals'] > 0)).mean()
            }
        
        print("\n📊 League Statistics:")
        for league, stats in results.items():
            print(f"\n   {league}:")
            print(f"      Matches:      {stats['total_matches']}")
            print(f"      Avg Goals:    {stats['avg_goals']:.2f}")
            print(f"      Over 2.5:     {stats['over_2_5_rate']*100:.1f}%")
        
        return results
    
    def _generate_visualizations(self, wf_results: Dict, timestamp: str):
        """Generate all visualization charts"""
        charts_dir = f"{self.output_dir}/charts_{timestamp}"
        os.makedirs(charts_dir, exist_ok=True)
        
        if 'raw_bets' not in wf_results or not wf_results['raw_bets']:
            print("   ⚠️  No bets to visualize")
            return
        
        df = pd.DataFrame(wf_results['raw_bets'])
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Equity Curve
        self._plot_equity_curve(df, charts_dir)
        
        # 2. Drawdown Chart
        self._plot_drawdown(df, charts_dir)
        
        # 3. Win Rate by Market
        self._plot_win_rate_by_market(df, charts_dir)
        
        # 4. ROI Distribution
        self._plot_roi_distribution(df, charts_dir)
        
        # 5. Window Performance
        if 'window_results' in wf_results:
            self._plot_window_performance(wf_results['window_results'], charts_dir)
        
        print(f"   ✅ Saved 5 charts to {charts_dir}/")
    
    def _plot_equity_curve(self, df: pd.DataFrame, charts_dir: str):
        """Plot bankroll over time"""
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(df)), df['bankroll'], linewidth=2, color='#2E86AB')
        plt.axhline(y=1000, color='red', linestyle='--', label='Initial Bankroll')
        plt.fill_between(range(len(df)), 1000, df['bankroll'], alpha=0.3, color='#2E86AB')
        plt.xlabel('Bet Number', fontsize=12)
        plt.ylabel('Bankroll ($)', fontsize=12)
        plt.title('Equity Curve - Bankroll Evolution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/equity_curve.png", dpi=300)
        plt.close()
    
    def _plot_drawdown(self, df: pd.DataFrame, charts_dir: str):
        """Plot drawdown percentage"""
        bankroll = df['bankroll']
        running_max = bankroll.expanding().max()
        drawdown = ((bankroll - running_max) / running_max * 100)
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(range(len(drawdown)), 0, drawdown, color='#A23B72', alpha=0.6)
        plt.plot(range(len(drawdown)), drawdown, color='#A23B72', linewidth=2)
        plt.xlabel('Bet Number', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.title('Drawdown Analysis - Risk Exposure', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/drawdown.png", dpi=300)
        plt.close()
    
    def _plot_win_rate_by_market(self, df: pd.DataFrame, charts_dir: str):
        """Plot win rate by market"""
        market_wins = df.groupby('market')['won'].agg(['sum', 'count', 'mean'])
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(market_wins)), market_wins['mean'] * 100, color='#18A558')
        plt.xticks(range(len(market_wins)), [m.replace('_', ' ').title() for m in market_wins.index], rotation=45)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.title('Win Rate by Market', fontsize=14, fontweight='bold')
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Break-even (50%)')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/win_rate_by_market.png", dpi=300)
        plt.close()
    
    def _plot_roi_distribution(self, df: pd.DataFrame, charts_dir: str):
        """Plot ROI distribution (profit per bet)"""
        plt.figure(figsize=(10, 6))
        plt.hist(df['profit'], bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        plt.xlabel('Profit per Bet ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Profit Distribution per Bet', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/roi_distribution.png", dpi=300)
        plt.close()
    
    def _plot_window_performance(self, window_results: List[Dict], charts_dir: str):
        """Plot performance by window"""
        df_windows = pd.DataFrame(window_results)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Window ROI
        ax1.bar(df_windows['window'], df_windows['roi'], color=['#18A558' if x > 0 else '#A23B72' for x in df_windows['roi']])
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Window Number', fontsize=12)
        ax1.set_ylabel('ROI (%)', fontsize=12)
        ax1.set_title('Window-by-Window ROI', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Cumulative bankroll
        ax2.plot(df_windows['window'], df_windows['bankroll'], marker='o', linewidth=2, color='#2E86AB')
        ax2.axhline(y=1000, color='red', linestyle='--', label='Initial Bankroll')
        ax2.fill_between(df_windows['window'], 1000, df_windows['bankroll'], alpha=0.3, color='#2E86AB')
        ax2.set_xlabel('Window Number', fontsize=12)
        ax2.set_ylabel('Bankroll ($)', fontsize=12)
        ax2.set_title('Cumulative Bankroll by Window', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/window_performance.png", dpi=300)
        plt.close()
    
    def _generate_report(self, timestamp: str) -> str:
        """Generate comprehensive text report"""
        report_path = f"{self.output_dir}/stress_test_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("🏋️  10,000+ MATCH STRESS TEST REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"System: telegramsoccer ML Betting System\n\n")
            
            # Data info
            f.write("-"*80 + "\n")
            f.write("📊 DATASET INFORMATION\n")
            f.write("-"*80 + "\n")
            data_info = self.results['data_info']
            f.write(f"Total Matches:   {data_info['total_matches']:,}\n")
            f.write(f"Date Range:      {data_info['date_range']}\n")
            f.write(f"Leagues:         {data_info['leagues']}\n")
            f.write(f"Unique Teams:    {data_info['teams']}\n\n")
            
            # Walk-forward results
            f.write("-"*80 + "\n")
            f.write("🔄 WALK-FORWARD BACKTEST (PRIMARY TEST)\n")
            f.write("-"*80 + "\n")
            wf = self.results['walk_forward']
            if 'summary' in wf:
                s = wf['summary']
                f.write(f"Total Bets:      {s['total_bets']}\n")
                f.write(f"Wins:            {s['wins']} ({s['win_rate']*100:.1f}%)\n")
                f.write(f"Losses:          {s['losses']}\n")
                f.write(f"Initial:         ${s['initial_bankroll']:,.2f}\n")
                f.write(f"Final:           ${s['final_bankroll']:,.2f}\n")
                f.write(f"Profit:          ${s['total_profit']:+,.2f}\n")
                f.write(f"ROI:             {s['roi']:+.2f}%\n\n")
                
                f.write("Risk Metrics:\n")
                r = wf['risk']
                f.write(f"  Max Drawdown:  {r['max_drawdown_pct']:.2f}%\n")
                f.write(f"  Sharpe Ratio:  {r['sharpe_ratio']:.3f}\n")
                f.write(f"  Sortino Ratio: {r['sortino_ratio']:.3f}\n\n")
            
            # Benchmarks
            f.write("-"*80 + "\n")
            f.write("📊 BENCHMARK COMPARISONS\n")
            f.write("-"*80 + "\n")
            
            f.write("\nML System vs Random Betting:\n")
            rand = self.results['random_benchmark']
            ml_roi = wf['summary']['roi'] if 'summary' in wf else 0
            f.write(f"  ML System:       {ml_roi:+.2f}% ROI\n")
            f.write(f"  Random:          {rand['roi']:+.2f}% ROI\n")
            f.write(f"  Advantage:       {ml_roi - rand['roi']:+.2f}%\n\n")
            
            f.write("ML System vs Favorite Betting:\n")
            fav = self.results['favorite_benchmark']
            f.write(f"  ML System:       {ml_roi:+.2f}% ROI\n")
            f.write(f"  Favorites:       {fav['roi']:+.2f}% ROI\n")
            f.write(f"  Advantage:       {ml_roi - fav['roi']:+.2f}%\n\n")
            
            # Verdict
            f.write("="*80 + "\n")
            f.write("🏆 FINAL VERDICT\n")
            f.write("="*80 + "\n")
            
            if ml_roi > 10:
                verdict = "✅ EXCELLENT - Strong profitable system"
            elif ml_roi > 5:
                verdict = "✅ GOOD - Profitable with room for improvement"
            elif ml_roi > 0:
                verdict = "⚠️  MARGINAL - Barely profitable, needs optimization"
            else:
                verdict = "❌ UNPROFITABLE - System needs major rework"
            
            f.write(f"{verdict}\n")
            f.write(f"\nTested on {data_info['total_matches']:,} historical matches\n")
            f.write(f"Outperforms random betting by {ml_roi - rand['roi']:+.2f}%\n")
            f.write(f"Outperforms favorite betting by {ml_roi - fav['roi']:+.2f}%\n\n")
            
            f.write("="*80 + "\n")
        
        return report_path


if __name__ == "__main__":
    print("\n🚀 Starting 10,000+ Match Stress Test...")
    
    suite = StressTestSuite()
    results = suite.run_full_stress_test()
    
    print("\n✅ All tests complete! Check output directory for reports and charts.")
