"""Professional Streamlit dashboard for betting system monitoring."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from src.core.database import DatabaseManager
from src.betting.engine import BettingEngine


st.set_page_config(
    page_title="TelegramSoccer Pro Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


class DashboardApp:
    """Main dashboard application."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.betting_engine = BettingEngine(self.db)
    
    def run(self):
        """Main dashboard logic."""
        st.title("⚽ TelegramSoccer Pro Dashboard")
        st.markdown("*Advanced AI-Powered Soccer Betting Analytics*")
        
        # Sidebar filters
        st.sidebar.header("📊 Filters")
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
        )
        
        league_filter = st.sidebar.multiselect(
            "Leagues",
            options=["All", "Premier League", "Bundesliga", "La Liga", "Serie A"],
            default=["All"],
        )
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Performance Overview",
            "🎯 Today's Tips",
            "🤖 Model Analytics",
            "💰 Bankroll Management",
            "📚 Learning Insights",
        ])
        
        with tab1:
            self._performance_overview(date_range)
        
        with tab2:
            self._todays_tips()
        
        with tab3:
            self._model_analytics(date_range)
        
        with tab4:
            self._bankroll_management()
        
        with tab5:
            self._learning_insights()
    
    def _performance_overview(self, date_range):
        """Tab 1: Performance KPIs."""
        st.header("📈 Performance Overview")
        
        # Fetch statistics
        stats = self.betting_engine.get_statistics()
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Win Rate",
                value=f"{stats['win_rate']:.1f}%",
                delta=f"{stats.get('win_rate_change', 0):.1f}% vs last week",
            )
        
        with col2:
            st.metric(
                label="ROI",
                value=f"{stats['roi']:.1f}%",
                delta=f"{stats.get('roi_change', 0):.1f}%",
            )
        
        with col3:
            st.metric(
                label="Total Profit",
                value=f"€{stats['total_profit']:.2f}",
                delta=f"€{stats.get('profit_change', 0):.2f}",
            )
        
        with col4:
            st.metric(
                label="Avg Odds",
                value=f"{stats['average_odds']:.2f}",
                delta=None,
            )
        
        # Profit/Loss Chart
        st.subheader("💰 Cumulative Profit/Loss")
        
        # Fetch historical data
        df_history = self._fetch_betting_history(date_range)
        
        if not df_history.empty:
            df_history['cumulative_pl'] = df_history['profit'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_history['date'],
                y=df_history['cumulative_pl'],
                mode='lines+markers',
                name='Cumulative P/L',
                line=dict(color='#00CC96', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 204, 150, 0.1)',
            ))
            
            fig.update_layout(
                height=400,
                hovermode='x unified',
                xaxis_title="Date",
                yaxis_title="Profit/Loss (€)",
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No betting history available for selected date range")
        
        # Win/Loss Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Win/Loss Distribution")
            
            fig = go.Figure(data=[go.Pie(
                labels=['Won', 'Lost'],
                values=[stats['bets_won'], stats['bets_placed'] - stats['bets_won']],
                marker=dict(colors=['#00CC96', '#EF553B']),
            )])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Performance by Market")
            
            market_stats = self._fetch_market_stats(date_range)
            
            fig = px.bar(
                market_stats,
                x='market',
                y='roi',
                color='win_rate',
                title="ROI by Market Type",
                color_continuous_scale='Viridis',
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def _todays_tips(self):
        """Tab 2: Today's betting tips."""
        st.header("🎯 Today's Professional Tips")
        
        # Fetch today's tips
        tips = self._fetch_todays_tips()
        
        if not tips:
            st.info("No tips generated for today yet. Check back at 9:00 AM UTC.")
            return
        
        st.success(f"**{len(tips)} Tips Available**")
        
        # Display tips in cards
        for i, tip in enumerate(tips, 1):
            with st.expander(f"**Tip #{i}**: {tip['match']} - {tip['market']} @ {tip['odds']}", expanded=(i <= 3)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**⚽ Match**: {tip['match']}")
                    st.markdown(f"**🎯 Market**: {tip['market']}")
                    st.markdown(f"**📊 Odds**: {tip['odds']}")
                    st.markdown(f"**💰 Recommended Stake**: {tip['stake_pct']:.1f}% (€{tip['stake_amount']:.2f})")
                    st.markdown(f"**🔮 Probability**: {tip['probability']:.1%}")
                    st.markdown(f"**✅ Confidence**: {tip['confidence']:.1%}")
                
                with col2:
                    # Value indicator
                    value_score = (tip['probability'] / (1 / tip['odds'])) - 1
                    st.metric("Value Score", f"{value_score:.1%}")
                    
                    # Expected value
                    ev = tip['probability'] * (tip['odds'] - 1) - (1 - tip['probability'])
                    st.metric("Expected Value", f"{ev:.2f}")
                
                st.markdown("---")
                st.markdown("**🔑 Key Factors:**")
                for factor in tip.get('key_factors', []):
                    st.markdown(f"- {factor}")
                
                st.markdown("---")
                st.markdown("**💭 Analysis:**")
                st.markdown(tip.get('reasoning', 'N/A'))
                
                # RAG context (if available)
                if tip.get('similar_mistakes'):
                    st.warning("⚠️ **Similar Past Mistakes Identified** - Analysis adjusted accordingly")
    
    def _model_analytics(self, date_range):
        """Tab 3: Model performance analytics."""
        st.header("🤖 Model Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧠 LLM Performance")
            llm_stats = self._fetch_model_stats('llm', date_range)
            
            st.metric("Accuracy", f"{llm_stats.get('accuracy', 0):.1%}")
            st.metric("Avg Confidence", f"{llm_stats.get('avg_confidence', 0):.1%}")
            st.metric("Calibration Error", f"{llm_stats.get('calibration_error', 0):.3f}")
            
            # LLM accuracy by league
            st.markdown("**Performance by League:**")
            league_acc = llm_stats.get('league_accuracy', {})
            for league, acc in league_acc.items():
                st.markdown(f"- {league}: {acc:.1%}")
        
        with col2:
            st.subheader("📈 XGBoost Performance")
            xgb_stats = self._fetch_model_stats('xgboost', date_range)
            
            st.metric("Accuracy", f"{xgb_stats.get('accuracy', 0):.1%}")
            st.metric("Precision", f"{xgb_stats.get('precision', 0):.1%}")
            st.metric("Recall", f"{xgb_stats.get('recall', 0):.1%}")
            
            # Feature importance
            st.markdown("**Top Features:**")
            feature_imp = xgb_stats.get('feature_importance', {})
            for feat, imp in list(feature_imp.items())[:5]:
                st.markdown(f"- {feat}: {imp:.3f}")
        
        # Ensemble weighting over time
        st.subheader("🎭 Ensemble Model Weighting")
        
        weight_history = self._fetch_ensemble_weights(date_range)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weight_history['date'],
            y=weight_history['llm_weight'],
            mode='lines',
            name='LLM Weight',
            stackgroup='one',
        ))
        fig.add_trace(go.Scatter(
            x=weight_history['date'],
            y=weight_history['xgboost_weight'],
            mode='lines',
            name='XGBoost Weight',
            stackgroup='one',
        ))
        
        fig.update_layout(
            height=300,
            yaxis_title="Weight",
            xaxis_title="Date",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # RL Agent Performance
        st.subheader("🎮 RL Agent Staking Performance")
        
        rl_stats = self._fetch_rl_stats(date_range)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Stake", f"{rl_stats.get('avg_stake_pct', 0):.1f}%")
        col2.metric("Bets Avoided", f"{rl_stats.get('bets_avoided', 0)}")
        col3.metric("ROI vs Fixed 2%", f"+{rl_stats.get('roi_improvement', 0):.1f}%")
    
    def _bankroll_management(self):
        """Tab 4: Bankroll management."""
        st.header("💰 Bankroll Management")
        
        # Current bankroll
        bankroll = self.betting_engine.get_bankroll()
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Current Bankroll", f"€{bankroll:.2f}")
        col2.metric("All-Time High", f"€{self.betting_engine.stats.get('ath_bankroll', 0):.2f}")
        col3.metric("Drawdown", f"{self.betting_engine.stats.get('max_drawdown', 0):.1f}%")
        
        # Stop-loss warning
        drawdown_pct = (1 - bankroll / self.betting_engine.stats.get('ath_bankroll', bankroll)) * 100
        
        if drawdown_pct >= 15:
            st.error("🚨 **STOP-LOSS TRIGGERED** - System paused until review")
        elif drawdown_pct >= 10:
            st.warning("⚠️ **Approaching Stop-Loss** - Exercise caution")
        else:
            st.success("✅ Bankroll healthy - within risk parameters")
        
        # Bankroll history chart
        st.subheader("📈 Bankroll Growth")
        
        bankroll_history = self._fetch_bankroll_history()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bankroll_history['date'],
            y=bankroll_history['balance'],
            mode='lines',
            name='Bankroll',
            line=dict(color='#636EFA', width=2),
        ))
        
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    def _learning_insights(self):
        """Tab 5: Continuous learning insights."""
        st.header("📚 Learning Insights")
        
        st.subheader("🔥 Recent Post-Mortems")
        
        recent_failures = self._fetch_recent_failures(limit=5)
        
        for failure in recent_failures:
            with st.expander(f"{failure['match']} - {failure['market']} (Lost)"):
                st.markdown(f"**Date**: {failure['date']}")
                st.markdown(f"**Odds**: {failure['odds']}")
                st.markdown(f"**Predicted Probability**: {failure['probability']:.1%}")
                st.markdown("---")
                st.markdown("**What Went Wrong:**")
                st.markdown(failure['post_mortem'])
        
        # RAG system stats
        st.subheader("🧠 RAG Memory Bank")
        
        rag_stats = self._fetch_rag_stats()
        
        col1, col2 = st.columns(2)
        col1.metric("Total Memories", rag_stats.get('total_memories', 0))
        col2.metric("Avg Retrieval Distance", f"{rag_stats.get('avg_distance', 0):.3f}")
        
        # Fine-tuning history
        st.subheader("🔧 Model Retraining History")
        
        retraining_log = self._fetch_retraining_log()
        
        st.dataframe(
            retraining_log,
            use_container_width=True,
            hide_index=True,
        )
    
    # Helper methods (fetch data from database)
    def _fetch_betting_history(self, date_range):
        """Fetch betting history for date range."""
        # Placeholder - query database
        return pd.DataFrame({
            'date': pd.date_range(start=date_range[0], end=date_range[1], freq='D'),
            'profit': [10, -5, 15, 20, -10, 5, 12, 8, -3, 18] * 3,
        })
    
    def _fetch_market_stats(self, date_range):
        """Fetch performance by market."""
        return pd.DataFrame({
            'market': ['Over 1.5', 'BTTS', 'Double Chance'],
            'roi': [12.5, 8.3, 15.2],
            'win_rate': [68, 62, 71],
        })
    
    def _fetch_todays_tips(self):
        """Fetch today's tips."""
        # Placeholder
        return []
    
    def _fetch_model_stats(self, model_name, date_range):
        """Fetch model statistics."""
        return {
            'accuracy': 0.68,
            'avg_confidence': 0.72,
            'calibration_error': 0.045,
            'league_accuracy': {
                'Bundesliga': 0.71,
                'Premier League': 0.65,
            },
        }
    
    def _fetch_ensemble_weights(self, date_range):
        """Fetch ensemble weight history."""
        return pd.DataFrame({
            'date': pd.date_range(start=date_range[0], end=date_range[1], freq='D'),
            'llm_weight': [0.6] * 30,
            'xgboost_weight': [0.4] * 30,
        })
    
    def _fetch_rl_stats(self, date_range):
        """Fetch RL agent stats."""
        return {
            'avg_stake_pct': 2.3,
            'bets_avoided': 12,
            'roi_improvement': 3.2,
        }
    
    def _fetch_bankroll_history(self):
        """Fetch bankroll history."""
        return pd.DataFrame({
            'date': pd.date_range(start='2026-01-01', periods=30, freq='D'),
            'balance': [1000 + i*10 for i in range(30)],
        })
    
    def _fetch_recent_failures(self, limit=5):
        """Fetch recent failed tips."""
        return []
    
    def _fetch_rag_stats(self):
        """Fetch RAG system statistics."""
        return {
            'total_memories': 127,
            'avg_distance': 0.234,
        }
    
    def _fetch_retraining_log(self):
        """Fetch retraining history."""
        return pd.DataFrame({
            'Date': ['2026-01-20', '2026-01-13', '2026-01-06'],
            'Model': ['Meta-Learner', 'RL Agent', 'XGBoost'],
            'Performance Before': ['62%', '58%', '64%'],
            'Performance After': ['68%', '65%', '67%'],
            'Samples Used': [234, 150, 189],
        })


if __name__ == "__main__":
    app = DashboardApp()
    app.run()
