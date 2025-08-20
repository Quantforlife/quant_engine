"""
Multi-Modal Regime-Switching Alpha Engine
Main entry point for the quantitative finance platform
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
from src.utils import setup_logging, ensure_output_dir

def main():
    """Main execution pipeline"""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Ensure output directory exists
    ensure_output_dir()
    
    logger.info("Starting Multi-Modal Regime-Switching Alpha Engine")
    
    try:
        # Initialize configuration
        config = Config()
        
        # Check for FRED API key
        fred_api_key = os.getenv('FRED_API_KEY')
        if not fred_api_key:
            logger.error("FRED_API_KEY environment variable not found. Please set it before running.")
            print("ERROR: FRED_API_KEY environment variable not found.")
            print("Please set your FRED API key: export FRED_API_KEY='your_key_here'")
            return
        
        # Step 1: Data Fetching
        logger.info("Step 1: Fetching market and economic data...")
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
        
        # Step 2: Regime Detection
        logger.info("Step 2: Detecting market regimes...")
        regime_detector = RegimeDetector()
        regime_data = regime_detector.detect_regimes(market_data, economic_data)
        
        # Step 3: Alpha Model Generation
        logger.info("Step 3: Generating alpha signals...")
        alpha_models = AlphaModels()
        alpha_signals = alpha_models.generate_signals(market_data, regime_data)
        
        # Step 4: RL Portfolio Management
        logger.info("Step 4: Training RL portfolio manager...")
        rl_manager = RLPortfolioManager()
        portfolio_weights = rl_manager.optimize_portfolio(alpha_signals, market_data)
        
        # Step 5: Risk Management
        logger.info("Step 5: Applying risk management...")
        risk_manager = RiskManager()
        adjusted_weights = risk_manager.apply_risk_controls(
            portfolio_weights, market_data, alpha_signals
        )
        
        # Step 6: Backtesting
        logger.info("Step 6: Running backtest...")
        backtester = Backtester()
        backtest_results = backtester.run_backtest(
            market_data, alpha_signals, adjusted_weights, regime_data
        )
        
        # Step 7: Trade Reporting
        logger.info("Step 7: Generating trade reports...")
        trade_reporter = TradeReporter()
        trade_reporter.generate_reports(backtest_results)
        
        # Display summary results
        logger.info("Pipeline completed successfully!")
        print("\n" + "="*60)
        print("MULTI-MODAL REGIME-SWITCHING ALPHA ENGINE - RESULTS")
        print("="*60)
        
        metrics = backtest_results.get('metrics', {})
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
        print(f"Sortino Ratio: {metrics.get('sortino_ratio', 'N/A'):.4f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.4f}")
        print(f"CAGR: {metrics.get('cagr', 'N/A'):.4f}")
        print(f"Total Trades: {metrics.get('total_trades', 'N/A')}")
        
        print(f"\nReports saved to: output/")
        print(f"Trade log: output/trade_report_TEST{trade_reporter.test_number}.csv")
        print("\nTo view the interactive dashboard, run: streamlit run app.py")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"ERROR: Pipeline failed - {str(e)}")
        raise

if __name__ == "__main__":
    main()
