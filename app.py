"""
Streamlit Dashboard for Multi-Modal Regime-Switching Alpha Engine
Interactive web interface for monitoring and visualizing the quant platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_fetcher import DataFetcher
from src.regime_detector import RegimeDetector
from src.alpha_models import AlphaModels
from src.rl_portfolio_manager import RLPortfolioManager
from src.risk_manager import RiskManager
from src.backtester import Backtester
from src.trade_reporter import TradeReporter
from src.config import Config
from src.utils import ensure_output_dir

# Page configuration
st.set_page_config(
    page_title="Multi-Modal Regime-Switching Alpha Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False
if 'results' not in st.session_state:
    st.session_state.results = None

def load_existing_data():
    """Load any existing output data"""
    output_files = []
    if os.path.exists('output'):
        output_files = [f for f in os.listdir('output') if f.endswith('.csv')]
    return output_files

def run_pipeline():
    """Execute the complete analysis pipeline"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize configuration
        config = Config()
        
        # Check for FRED API key
        fred_api_key = os.getenv('FRED_API_KEY')
        if not fred_api_key:
            st.error("FRED_API_KEY environment variable not found. Please set it in your environment.")
            return None
        
        # Ensure output directory
        ensure_output_dir()
        
        # Step 1: Data Fetching (20%)
        status_text.text("Fetching market and economic data...")
        progress_bar.progress(20)
        
        data_fetcher = DataFetcher(fred_api_key)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.LOOKBACK_DAYS)
        
        market_data = data_fetcher.fetch_market_data(
            symbols=config.MARKET_SYMBOLS,
            start_date=start_date,
            end_date=end_date
        )
        
        economic_data = data_fetcher.fetch_economic_data(
            indicators=config.ECONOMIC_INDICATORS,
            start_date=start_date,
            end_date=end_date
        )
        
        # Step 2: Regime Detection (40%)
        status_text.text("Detecting market regimes...")
        progress_bar.progress(40)
        
        regime_detector = RegimeDetector()
        regime_data = regime_detector.detect_regimes(market_data, economic_data)
        
        # Step 3: Alpha Models (60%)
        status_text.text("Generating alpha signals...")
        progress_bar.progress(60)
        
        alpha_models = AlphaModels()
        alpha_signals = alpha_models.generate_signals(market_data, regime_data)
        
        # Step 4: RL Portfolio Management (80%)
        status_text.text("Optimizing portfolio with RL...")
        progress_bar.progress(80)
        
        rl_manager = RLPortfolioManager()
        portfolio_weights = rl_manager.optimize_portfolio(alpha_signals, market_data)
        
        # Step 5: Risk Management & Backtesting (90%)
        status_text.text("Applying risk management and backtesting...")
        progress_bar.progress(90)
        
        risk_manager = RiskManager()
        adjusted_weights = risk_manager.apply_risk_controls(
            portfolio_weights, market_data, alpha_signals
        )
        
        backtester = Backtester()
        backtest_results = backtester.run_backtest(
            market_data, alpha_signals, adjusted_weights, regime_data
        )
        
        # Step 6: Generate Reports (100%)
        status_text.text("Generating reports...")
        progress_bar.progress(100)
        
        trade_reporter = TradeReporter()
        trade_reporter.generate_reports(backtest_results)
        
        status_text.text("Pipeline completed successfully!")
        
        return {
            'market_data': market_data,
            'economic_data': economic_data,
            'regime_data': regime_data,
            'alpha_signals': alpha_signals,
            'portfolio_weights': adjusted_weights,
            'backtest_results': backtest_results,
            'test_number': trade_reporter.test_number
        }
        
    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")
        return None

def display_overview_tab(results):
    """Display overview metrics and key charts"""
    
    if results is None:
        st.info("Run the pipeline to see overview metrics.")
        return
    
    metrics = results['backtest_results'].get('metrics', {})
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Sharpe Ratio",
            value=f"{metrics.get('sharpe_ratio', 0):.4f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Sortino Ratio",
            value=f"{metrics.get('sortino_ratio', 0):.4f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Max Drawdown",
            value=f"{metrics.get('max_drawdown', 0):.4f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="CAGR",
            value=f"{metrics.get('cagr', 0):.4f}",
            delta=None
        )
    
    # Portfolio equity curve
    st.subheader("Portfolio Equity Curve")
    
    equity_curve = results['backtest_results'].get('equity_curve', pd.Series())
    if not equity_curve.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity curve data available.")

def display_regime_tab(results):
    """Display regime detection analysis"""
    
    if results is None:
        st.info("Run the pipeline to see regime analysis.")
        return
    
    st.subheader("Market Regime Detection")
    
    regime_data = results.get('regime_data', {})
    market_data = results.get('market_data', {})
    
    if regime_data and market_data:
        # Create regime visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Market Prices with Regime Overlay', 'Regime Probabilities'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Market prices with regime coloring
        for symbol in market_data.keys():
            prices = market_data[symbol]['Close']
            fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=prices.values,
                    mode='lines',
                    name=f'{symbol} Price',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Regime probabilities
        if 'regime_probs' in regime_data:
            regime_probs = regime_data['regime_probs']
            for i, regime_name in enumerate(['Bullish', 'Bearish', 'Volatile']):
                if i < regime_probs.shape[1]:
                    fig.add_trace(
                        go.Scatter(
                            x=regime_probs.index,
                            y=regime_probs.iloc[:, i],
                            mode='lines',
                            name=f'{regime_name} Probability',
                            line=dict(width=1.5)
                        ),
                        row=2, col=1
                    )
        
        fig.update_layout(height=600, title="Market Regime Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Regime statistics
        st.subheader("Regime Statistics")
        if 'regime_labels' in regime_data:
            regime_labels = regime_data['regime_labels']
            regime_counts = pd.Series(regime_labels).value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Regime Distribution:**")
                for regime, count in regime_counts.items():
                    percentage = (count / len(regime_labels)) * 100
                    st.write(f"Regime {regime}: {count} days ({percentage:.1f}%)")
            
            with col2:
                # Regime pie chart
                fig_pie = px.pie(
                    values=regime_counts.values,
                    names=[f'Regime {i}' for i in regime_counts.index],
                    title="Regime Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    
    else:
        st.info("No regime data available.")

def display_alpha_tab(results):
    """Display alpha model signals and performance"""
    
    if results is None:
        st.info("Run the pipeline to see alpha model analysis.")
        return
    
    st.subheader("Alpha Model Signals")
    
    alpha_signals = results.get('alpha_signals', {})
    
    if alpha_signals:
        # Display signals for each model
        for model_name, signals in alpha_signals.items():
            st.write(f"**{model_name.replace('_', ' ').title()} Model**")
            
            if isinstance(signals, pd.DataFrame) and not signals.empty:
                # Plot signals
                fig = go.Figure()
                
                for column in signals.columns:
                    fig.add_trace(go.Scatter(
                        x=signals.index,
                        y=signals[column],
                        mode='lines',
                        name=f'{column}',
                        line=dict(width=1.5)
                    ))
                
                fig.update_layout(
                    title=f"{model_name.replace('_', ' ').title()} Signals",
                    xaxis_title="Date",
                    yaxis_title="Signal Strength",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No signal data available for {model_name}")
    else:
        st.info("No alpha signals available.")

def display_portfolio_tab(results):
    """Display portfolio allocation and performance"""
    
    if results is None:
        st.info("Run the pipeline to see portfolio analysis.")
        return
    
    st.subheader("Portfolio Allocation")
    
    portfolio_weights = results.get('portfolio_weights', {})
    backtest_results = results.get('backtest_results', {})
    
    if portfolio_weights:
        # Portfolio weights over time
        if isinstance(portfolio_weights, pd.DataFrame):
            fig = go.Figure()
            
            for column in portfolio_weights.columns:
                fig.add_trace(go.Scatter(
                    x=portfolio_weights.index,
                    y=portfolio_weights[column],
                    mode='lines',
                    name=column,
                    line=dict(width=2),
                    stackgroup='one'
                ))
            
            fig.update_layout(
                title="Portfolio Weights Over Time",
                xaxis_title="Date",
                yaxis_title="Allocation Weight",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    if 'metrics' in backtest_results:
        metrics = backtest_results['metrics']
        
        st.subheader("Detailed Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Return Metrics:**")
            st.write(f"Total Return: {metrics.get('total_return', 'N/A'):.4f}")
            st.write(f"CAGR: {metrics.get('cagr', 'N/A'):.4f}")
            st.write(f"Volatility: {metrics.get('volatility', 'N/A'):.4f}")
            
        with col2:
            st.write("**Risk Metrics:**")
            st.write(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
            st.write(f"Sortino Ratio: {metrics.get('sortino_ratio', 'N/A'):.4f}")
            st.write(f"Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.4f}")

def display_trades_tab(results):
    """Display trade log and analysis"""
    
    st.subheader("Trade Log and Analysis")
    
    # Load existing trade reports
    output_files = load_existing_data()
    trade_files = [f for f in output_files if f.startswith('trade_report_')]
    
    if trade_files:
        # Select trade report
        selected_file = st.selectbox("Select Trade Report:", trade_files)
        
        if selected_file:
            try:
                trade_df = pd.read_csv(f'output/{selected_file}')
                
                # Display trade statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Trades", len(trade_df))
                
                with col2:
                    profitable_trades = len(trade_df[trade_df['PnL'] > 0])
                    win_rate = profitable_trades / len(trade_df) if len(trade_df) > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.2%}")
                
                with col3:
                    total_pnl = trade_df['PnL'].sum()
                    st.metric("Total P&L", f"{total_pnl:.4f}")
                
                # Display trade table
                st.subheader("Recent Trades")
                st.dataframe(trade_df.tail(20), use_container_width=True)
                
                # P&L distribution
                if not trade_df.empty:
                    fig = px.histogram(
                        trade_df,
                        x='PnL',
                        title="P&L Distribution",
                        nbins=30
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading trade report: {str(e)}")
    else:
        st.info("No trade reports found. Run the pipeline to generate trade data.")

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">Multi-Modal Regime-Switching Alpha Engine</div>', 
                unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.title("Control Panel")
    
    # API Key status
    fred_api_key = os.getenv('FRED_API_KEY')
    if fred_api_key:
        st.sidebar.success("✅ FRED API Key Found")
    else:
        st.sidebar.error("❌ FRED API Key Missing")
        st.sidebar.info("Set FRED_API_KEY environment variable")
    
    # Run pipeline button
    if st.sidebar.button("🚀 Run All Tests", type="primary", disabled=not fred_api_key):
        with st.spinner("Running complete analysis pipeline..."):
            results = run_pipeline()
            if results:
                st.session_state.results = results
                st.session_state.pipeline_run = True
                st.sidebar.success("Pipeline completed successfully!")
            else:
                st.sidebar.error("Pipeline failed!")
    
    # Display existing output files
    st.sidebar.subheader("Output Files")
    output_files = load_existing_data()
    if output_files:
        for file in output_files:
            st.sidebar.text(f"📄 {file}")
    else:
        st.sidebar.info("No output files yet")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔄 Regime Detection", 
        "⚡ Alpha Models", 
        "📈 Portfolio", 
        "💰 Trades"
    ])
    
    results = st.session_state.results
    
    with tab1:
        display_overview_tab(results)
    
    with tab2:
        display_regime_tab(results)
    
    with tab3:
        display_alpha_tab(results)
    
    with tab4:
        display_portfolio_tab(results)
    
    with tab5:
        display_trades_tab(results)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Multi-Modal Regime-Switching Alpha Engine | "
        "Built with Streamlit & Python"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
