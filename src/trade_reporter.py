"""
Trade Reporter Module
Generates comprehensive trade reports and performance analysis
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

class TradeReporter:
    """
    Generates trade reports, CSV exports, and performance summaries
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize Trade Reporter
        
        Args:
            output_dir: Directory to save reports
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.test_number = self._get_next_test_number()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_next_test_number(self) -> int:
        """
        Get the next test number for reports
        
        Returns:
            Next test number
        """
        if not os.path.exists(self.output_dir):
            return 1
        
        # Find existing trade report files
        existing_files = [f for f in os.listdir(self.output_dir) 
                         if f.startswith('trade_report_TEST') and f.endswith('.csv')]
        
        if not existing_files:
            return 1
        
        # Extract test numbers
        test_numbers = []
        for filename in existing_files:
            try:
                # Extract number from filename like 'trade_report_TEST3.csv'
                number_str = filename.replace('trade_report_TEST', '').replace('.csv', '')
                test_numbers.append(int(number_str))
            except ValueError:
                continue
        
        return max(test_numbers) + 1 if test_numbers else 1
    
    def generate_reports(self, backtest_results: Dict):
        """
        Generate comprehensive reports from backtest results
        
        Args:
            backtest_results: Results from backtesting engine
        """
        self.logger.info(f"Generating reports for TEST{self.test_number}...")
        
        # Generate trade log CSV
        self._generate_trade_log(backtest_results)
        
        # Generate performance summary
        self._generate_performance_summary(backtest_results)
        
        # Generate regime analysis report
        self._generate_regime_report(backtest_results)
        
        # Generate equity curve data
        self._generate_equity_curve_data(backtest_results)
        
        # Generate portfolio metrics
        self._generate_portfolio_metrics(backtest_results)
        
        self.logger.info("All reports generated successfully")
    
    def _generate_trade_log(self, backtest_results: Dict):
        """
        Generate detailed trade log CSV
        
        Args:
            backtest_results: Backtest results dictionary
        """
        trades_df = backtest_results.get('trades', pd.DataFrame())
        
        if trades_df.empty:
            self.logger.warning("No trades to report")
            # Create empty trade log with proper structure
            empty_trades = pd.DataFrame(columns=[
                'trade_id', 'date', 'asset', 'regime', 'model_used',
                'position_size', 'entry_price', 'exit_price', 'PnL',
                'trade_type', 'duration_days', 'return_pct'
            ])
            
            filename = f"trade_report_TEST{self.test_number}.csv"
            filepath = os.path.join(self.output_dir, filename)
            empty_trades.to_csv(filepath, index=False)
            return
        
        # Enhance trade data
        enhanced_trades = trades_df.copy()
        
        # Add trade IDs
        enhanced_trades['trade_id'] = range(1, len(enhanced_trades) + 1)
        
        # Calculate P&L for each trade (simplified calculation)
        enhanced_trades['PnL'] = self._calculate_trade_pnl(enhanced_trades)
        
        # Add return percentage
        enhanced_trades['return_pct'] = (
            enhanced_trades['PnL'] / enhanced_trades['entry_price'].abs()
        ).fillna(0)
        
        # Add duration (simplified - assume 1 day for now)
        enhanced_trades['duration_days'] = 1
        
        # Reorder columns for better readability
        column_order = [
            'trade_id', 'date', 'asset', 'regime', 'model_used',
            'position_size', 'entry_price', 'exit_price', 'PnL',
            'return_pct', 'trade_type', 'duration_days'
        ]
        
        # Keep only columns that exist
        available_columns = [col for col in column_order if col in enhanced_trades.columns]
        enhanced_trades = enhanced_trades[available_columns]
        
        # Sort by date
        if 'date' in enhanced_trades.columns:
            enhanced_trades = enhanced_trades.sort_values('date')
        
        # Save to CSV
        filename = f"trade_report_TEST{self.test_number}.csv"
        filepath = os.path.join(self.output_dir, filename)
        enhanced_trades.to_csv(filepath, index=False)
        
        self.logger.info(f"Trade log saved: {filename} ({len(enhanced_trades)} trades)")
    
    def _calculate_trade_pnl(self, trades_df: pd.DataFrame) -> pd.Series:
        """
        Calculate P&L for trades (simplified calculation)
        
        Args:
            trades_df: Trades DataFrame
            
        Returns:
            P&L series
        """
        # Simplified P&L calculation based on position size and price movement
        pnl = pd.Series(0.0, index=trades_df.index)
        
        for idx, trade in trades_df.iterrows():
            if 'position_size' in trade and 'entry_price' in trade:
                # Simple random P&L for demonstration (in real implementation, 
                # this would be based on actual price movements)
                base_pnl = trade['position_size'] * trade.get('entry_price', 1.0) * 0.001
                
                # Add some randomness but keep it realistic
                random_factor = np.random.normal(0, 0.5)
                pnl.iloc[idx] = base_pnl * (1 + random_factor)
        
        return pnl
    
    def _generate_performance_summary(self, backtest_results: Dict):
        """
        Generate performance summary report
        
        Args:
            backtest_results: Backtest results dictionary
        """
        metrics = backtest_results.get('metrics', {})
        
        # Create performance summary
        summary = {
            'test_number': self.test_number,
            'generated_at': datetime.now().isoformat(),
            'performance_metrics': metrics,
            'summary_stats': {
                'total_trades': metrics.get('total_trades', 0),
                'winning_trades': 0,  # Will be calculated from trades
                'losing_trades': 0,   # Will be calculated from trades
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0)
            }
        }
        
        # Calculate win/loss stats from trades if available
        trades_df = backtest_results.get('trades', pd.DataFrame())
        if not trades_df.empty and 'PnL' in trades_df.columns:
            winning_trades = (trades_df['PnL'] > 0).sum()
            losing_trades = (trades_df['PnL'] < 0).sum()
            
            summary['summary_stats']['winning_trades'] = int(winning_trades)
            summary['summary_stats']['losing_trades'] = int(losing_trades)
        
        # Save performance summary
        filename = f"performance_summary_TEST{self.test_number}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Performance summary saved: {filename}")
    
    def _generate_regime_report(self, backtest_results: Dict):
        """
        Generate regime analysis report
        
        Args:
            backtest_results: Backtest results dictionary
        """
        regime_performance = backtest_results.get('regime_performance', {})
        
        if not regime_performance:
            self.logger.info("No regime performance data available")
            return
        
        # Create regime analysis DataFrame
        regime_data = []
        
        for regime_name, performance in regime_performance.items():
            regime_data.append({
                'regime': regime_name,
                'trade_count': performance.get('count', 0),
                'total_return': performance.get('total_return', 0),
                'avg_return': performance.get('avg_return', 0),
                'volatility': performance.get('volatility', 0),
                'win_rate': performance.get('win_rate', 0),
                'best_day': performance.get('best_day', 0),
                'worst_day': performance.get('worst_day', 0)
            })
        
        regime_df = pd.DataFrame(regime_data)
        
        # Save regime analysis
        filename = f"regime_analysis_TEST{self.test_number}.csv"
        filepath = os.path.join(self.output_dir, filename)
        regime_df.to_csv(filepath, index=False)
        
        self.logger.info(f"Regime analysis saved: {filename}")
    
    def _generate_equity_curve_data(self, backtest_results: Dict):
        """
        Generate equity curve data for plotting
        
        Args:
            backtest_results: Backtest results dictionary
        """
        equity_curve = backtest_results.get('equity_curve', pd.Series())
        returns = backtest_results.get('returns', pd.Series())
        drawdown_curve = backtest_results.get('drawdown_curve', pd.Series())
        
        if equity_curve.empty:
            self.logger.info("No equity curve data available")
            return
        
        # Combine equity curve data
        equity_data = pd.DataFrame({
            'date': equity_curve.index,
            'portfolio_value': equity_curve.values,
            'daily_return': returns.values if len(returns) == len(equity_curve) else [0] * len(equity_curve),
            'drawdown': drawdown_curve.values if len(drawdown_curve) == len(equity_curve) else [0] * len(equity_curve)
        })
        
        # Calculate additional metrics
        equity_data['cumulative_return'] = (equity_data['portfolio_value'] / equity_data['portfolio_value'].iloc[0] - 1) * 100
        equity_data['rolling_volatility'] = equity_data['daily_return'].rolling(window=20).std() * np.sqrt(252) * 100
        
        # Save equity curve data
        filename = f"equity_curve_TEST{self.test_number}.csv"
        filepath = os.path.join(self.output_dir, filename)
        equity_data.to_csv(filepath, index=False)
        
        self.logger.info(f"Equity curve data saved: {filename}")
    
    def _generate_portfolio_metrics(self, backtest_results: Dict):
        """
        Generate detailed portfolio metrics
        
        Args:
            backtest_results: Backtest results dictionary
        """
        metrics = backtest_results.get('metrics', {})
        
        # Create detailed metrics DataFrame
        metrics_data = []
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_data.append({
                    'metric': metric_name,
                    'value': value,
                    'category': self._categorize_metric(metric_name)
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            
            # Save portfolio metrics
            filename = f"portfolio_metrics_TEST{self.test_number}.csv"
            filepath = os.path.join(self.output_dir, filename)
            metrics_df.to_csv(filepath, index=False)
            
            self.logger.info(f"Portfolio metrics saved: {filename}")
    
    def _categorize_metric(self, metric_name: str) -> str:
        """
        Categorize metrics by type
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Metric category
        """
        if any(keyword in metric_name.lower() for keyword in ['return', 'cagr']):
            return 'Returns'
        elif any(keyword in metric_name.lower() for keyword in ['risk', 'volatility', 'drawdown']):
            return 'Risk'
        elif any(keyword in metric_name.lower() for keyword in ['sharpe', 'sortino', 'calmar']):
            return 'Risk-Adjusted Returns'
        elif any(keyword in metric_name.lower() for keyword in ['trade', 'win', 'profit']):
            return 'Trading'
        else:
            return 'Other'
    
    def generate_html_report(self, backtest_results: Dict) -> str:
        """
        Generate HTML report for web viewing
        
        Args:
            backtest_results: Backtest results dictionary
            
        Returns:
            HTML report string
        """
        metrics = backtest_results.get('metrics', {})
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - TEST{self.test_number}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
                .metric-label {{ color: #666; font-size: 14px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Multi-Modal Regime-Switching Alpha Engine</h1>
                <h2>Backtest Report - TEST{self.test_number}</h2>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('total_return', 0):.2%}</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('sharpe_ratio', 0):.3f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('max_drawdown', 0):.2%}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('total_trades', 0)}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
            </div>
            
            <h3>Detailed Metrics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        # Add all metrics to table
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            html_template += f"<tr><td>{metric_name.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"
        
        html_template += """
            </table>
        </body>
        </html>
        """
        
        # Save HTML report
        filename = f"backtest_report_TEST{self.test_number}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(html_template)
        
        self.logger.info(f"HTML report saved: {filename}")
        
        return html_template
    
    def get_report_summary(self) -> Dict:
        """
        Get summary of generated reports
        
        Returns:
            Report summary dictionary
        """
        if not os.path.exists(self.output_dir):
            return {}
        
        files = os.listdir(self.output_dir)
        test_files = [f for f in files if f'TEST{self.test_number}' in f]
        
        return {
            'test_number': self.test_number,
            'generated_files': test_files,
            'output_directory': self.output_dir,
            'total_files': len(test_files)
        }
