"""
Backtesting Module
Comprehensive backtesting engine for portfolio strategies
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class Backtester:
    """
    Comprehensive backtesting engine for portfolio strategies
    """
    
    def __init__(self, initial_capital: float = 100000, transaction_cost: float = 0.001,
                 slippage: float = 0.0001, margin_rate: float = 0.02):
        """
        Initialize Backtester
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost per trade (as percentage)
            slippage: Market impact slippage
            margin_rate: Margin interest rate (if leveraged)
        """
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.margin_rate = margin_rate
        
        # Results storage
        self.trades = []
        self.performance_metrics = {}
        
    def run_backtest(self, market_data: Dict[str, pd.DataFrame], 
                    alpha_signals: Dict[str, pd.DataFrame],
                    portfolio_weights: pd.DataFrame,
                    regime_data: Dict) -> Dict:
        """
        Run comprehensive backtest
        
        Args:
            market_data: Market data dictionary
            alpha_signals: Alpha model signals
            portfolio_weights: Portfolio allocation weights
            regime_data: Market regime information
            
        Returns:
            Backtest results dictionary
        """
        self.logger.info("Starting comprehensive backtest...")
        
        # Reset results
        self.trades = []
        self.performance_metrics = {}
        
        # Prepare data
        returns_data, prices_data = self._prepare_backtest_data(market_data)
        
        if returns_data.empty or portfolio_weights.empty:
            self.logger.warning("Insufficient data for backtesting")
            return self._create_empty_results()
        
        # Align all data
        aligned_data = self._align_backtest_data(
            returns_data, prices_data, portfolio_weights, regime_data
        )
        
        if not aligned_data:
            self.logger.warning("Data alignment failed")
            return self._create_empty_results()
        
        # Run simulation
        simulation_results = self._run_simulation(aligned_data, alpha_signals)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(simulation_results)
        
        # Generate results
        results = {
            'equity_curve': simulation_results['equity_curve'],
            'returns': simulation_results['portfolio_returns'],
            'positions': simulation_results['positions'],
            'trades': pd.DataFrame(self.trades),
            'metrics': performance_metrics,
            'drawdown_curve': simulation_results['drawdown_curve'],
            'regime_performance': self._analyze_regime_performance(
                simulation_results, aligned_data.get('regime_labels')
            )
        }
        
        self.logger.info("Backtest completed successfully")
        return results
    
    def _prepare_backtest_data(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare returns and price data for backtesting
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Tuple of (returns_data, prices_data)
        """
        returns_list = []
        prices_list = []
        
        for symbol, data in market_data.items():
            if data.empty:
                continue
            
            if 'Returns' in data.columns:
                returns_list.append(data['Returns'].rename(symbol))
            
            if 'Close' in data.columns:
                prices_list.append(data['Close'].rename(symbol))
        
        returns_df = pd.concat(returns_list, axis=1, join='inner') if returns_list else pd.DataFrame()
        prices_df = pd.concat(prices_list, axis=1, join='inner') if prices_list else pd.DataFrame()
        
        return returns_df.fillna(0), prices_df.fillna(method='ffill')
    
    def _align_backtest_data(self, returns_data: pd.DataFrame, prices_data: pd.DataFrame,
                            portfolio_weights: pd.DataFrame, regime_data: Dict) -> Dict:
        """
        Align all data sources for backtesting
        
        Args:
            returns_data: Returns DataFrame
            prices_data: Prices DataFrame  
            portfolio_weights: Portfolio weights DataFrame
            regime_data: Regime information
            
        Returns:
            Aligned data dictionary
        """
        # Find common date range
        date_indices = [returns_data.index, prices_data.index, portfolio_weights.index]
        date_indices = [idx for idx in date_indices if len(idx) > 0]
        
        if not date_indices:
            return {}
        
        # Get intersection of all dates
        common_dates = date_indices[0]
        for idx in date_indices[1:]:
            common_dates = common_dates.intersection(idx)
        
        if len(common_dates) < 10:  # Minimum data requirement
            self.logger.warning(f"Insufficient common dates: {len(common_dates)}")
            return {}
        
        # Align all data
        aligned_data = {
            'dates': common_dates,
            'returns': returns_data.loc[common_dates] if not returns_data.empty else pd.DataFrame(),
            'prices': prices_data.loc[common_dates] if not prices_data.empty else pd.DataFrame(),
            'weights': portfolio_weights.loc[common_dates]
        }
        
        # Add regime data if available
        if regime_data and 'regime_labels' in regime_data:
            regime_labels = regime_data['regime_labels']
            if len(regime_labels) >= len(common_dates):
                aligned_data['regime_labels'] = regime_labels[:len(common_dates)]
            else:
                # Extend regime labels if needed
                extended_regimes = np.zeros(len(common_dates))
                extended_regimes[:len(regime_labels)] = regime_labels
                aligned_data['regime_labels'] = extended_regimes
        
        return aligned_data
    
    def _run_simulation(self, aligned_data: Dict, alpha_signals: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run portfolio simulation
        
        Args:
            aligned_data: Aligned data dictionary
            alpha_signals: Alpha model signals
            
        Returns:
            Simulation results dictionary
        """
        dates = aligned_data['dates']
        returns = aligned_data['returns']
        prices = aligned_data['prices']
        weights = aligned_data['weights']
        
        # Initialize tracking variables
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = pd.DataFrame(0.0, index=dates, columns=returns.columns if not returns.empty else ['dummy'])
        
        equity_curve = []
        portfolio_returns = []
        drawdown_curve = []
        peak_value = self.initial_capital
        
        prev_weights = pd.Series(0.0, index=weights.columns)
        
        for i, date in enumerate(dates):
            current_weights = weights.loc[date]
            
            # Calculate portfolio return for the day
            if not returns.empty and i > 0:
                # Map weights to assets (simplified mapping)
                asset_returns = returns.loc[date]
                
                if len(asset_returns) > 0:
                    # Use first asset as portfolio proxy (simplified)
                    daily_return = current_weights.mean() * asset_returns.iloc[0]
                else:
                    daily_return = 0.0
            else:
                daily_return = 0.0
            
            # Apply transaction costs
            weight_changes = (current_weights - prev_weights).abs().sum()
            transaction_costs = weight_changes * self.transaction_cost * portfolio_value
            
            # Update portfolio value
            portfolio_value = portfolio_value * (1 + daily_return) - transaction_costs
            
            # Record trades
            if weight_changes > 0.001:  # Minimum trade threshold
                self._record_trades(date, prev_weights, current_weights, 
                                  prices.loc[date] if not prices.empty else pd.Series(),
                                  alpha_signals, aligned_data.get('regime_labels'))
            
            # Track performance
            equity_curve.append(portfolio_value)
            portfolio_returns.append(daily_return)
            
            # Calculate drawdown
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            
            drawdown = (peak_value - portfolio_value) / peak_value
            drawdown_curve.append(drawdown)
            
            prev_weights = current_weights.copy()
        
        return {
            'equity_curve': pd.Series(equity_curve, index=dates),
            'portfolio_returns': pd.Series(portfolio_returns, index=dates),
            'positions': positions,
            'drawdown_curve': pd.Series(drawdown_curve, index=dates)
        }
    
    def _record_trades(self, date: pd.Timestamp, prev_weights: pd.Series, 
                      current_weights: pd.Series, prices: pd.Series,
                      alpha_signals: Dict[str, pd.DataFrame], regime_labels: Optional[np.ndarray]):
        """
        Record individual trades
        
        Args:
            date: Trade date
            prev_weights: Previous weights
            current_weights: New weights
            prices: Current prices
            alpha_signals: Alpha signals for context
            regime_labels: Current regime labels
        """
        # Determine current regime
        current_regime = 0  # Default
        if regime_labels is not None and len(regime_labels) > len(self.trades):
            current_regime = regime_labels[len(self.trades)]
        
        # Determine dominant alpha model
        dominant_model = "momentum"  # Default
        if alpha_signals:
            # Find model with strongest signal
            max_signal_strength = 0
            for model_name, signals in alpha_signals.items():
                if not signals.empty and date in signals.index:
                    signal_strength = signals.loc[date].abs().sum()
                    if signal_strength > max_signal_strength:
                        max_signal_strength = signal_strength
                        dominant_model = model_name
        
        # Record trades for each weight change
        for asset in current_weights.index:
            weight_change = current_weights[asset] - prev_weights.get(asset, 0)
            
            if abs(weight_change) > 0.001:  # Minimum trade threshold
                entry_price = prices.get(asset, 1.0) if not prices.empty else 1.0
                
                trade_record = {
                    'date': date,
                    'asset': asset,
                    'regime': current_regime,
                    'model_used': dominant_model,
                    'position_size': weight_change,
                    'entry_price': entry_price,
                    'exit_price': entry_price,  # Simplified for now
                    'PnL': 0,  # Will be calculated later
                    'trade_type': 'BUY' if weight_change > 0 else 'SELL'
                }
                
                self.trades.append(trade_record)
    
    def _calculate_performance_metrics(self, simulation_results: Dict) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            simulation_results: Simulation results dictionary
            
        Returns:
            Performance metrics dictionary
        """
        equity_curve = simulation_results['equity_curve']
        returns = simulation_results['portfolio_returns']
        drawdown_curve = simulation_results['drawdown_curve']
        
        if equity_curve.empty or returns.empty:
            return {}
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        
        # Annualized metrics
        trading_days = len(returns)
        years = trading_days / 252
        
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = excess_returns / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown metrics
        max_drawdown = drawdown_curve.max() if not drawdown_curve.empty else 0
        
        # Calmar ratio
        calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0
        
        # Win rate
        winning_days = (returns > 0).sum()
        total_trading_days = len(returns)
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0
        
        # Average win/loss
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
        
        # Profit factor
        total_wins = winning_returns.sum() if len(winning_returns) > 0 else 0
        total_losses = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'trading_days': trading_days,
            'final_capital': equity_curve.iloc[-1],
            'peak_capital': equity_curve.max()
        }
        
        return metrics
    
    def _analyze_regime_performance(self, simulation_results: Dict, 
                                   regime_labels: Optional[np.ndarray]) -> Dict:
        """
        Analyze performance by market regime
        
        Args:
            simulation_results: Simulation results
            regime_labels: Regime labels array
            
        Returns:
            Regime performance analysis
        """
        if regime_labels is None:
            return {}
        
        returns = simulation_results['portfolio_returns']
        
        if returns.empty or len(regime_labels) != len(returns):
            return {}
        
        regime_performance = {}
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) == 0:
                continue
            
            regime_metrics = {
                'count': len(regime_returns),
                'total_return': regime_returns.sum(),
                'avg_return': regime_returns.mean(),
                'volatility': regime_returns.std() * np.sqrt(252),
                'win_rate': (regime_returns > 0).sum() / len(regime_returns),
                'best_day': regime_returns.max(),
                'worst_day': regime_returns.min()
            }
            
            regime_performance[f'regime_{regime}'] = regime_metrics
        
        return regime_performance
    
    def _create_empty_results(self) -> Dict:
        """Create empty results structure when backtest fails"""
        return {
            'equity_curve': pd.Series(),
            'returns': pd.Series(),
            'positions': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'metrics': {},
            'drawdown_curve': pd.Series(),
            'regime_performance': {}
        }
    
    def generate_performance_report(self, results: Dict) -> str:
        """
        Generate formatted performance report
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            Formatted report string
        """
        if not results or 'metrics' not in results:
            return "No backtest results available."
        
        metrics = results['metrics']
        
        report = [
            "="*60,
            "BACKTEST PERFORMANCE REPORT",
            "="*60,
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Final Capital: ${metrics.get('final_capital', 0):,.2f}",
            f"Total Return: {metrics.get('total_return', 0):.2%}",
            f"CAGR: {metrics.get('cagr', 0):.2%}",
            "",
            "RISK METRICS:",
            f"Volatility: {metrics.get('volatility', 0):.2%}",
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}",
            f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}",
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
            f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.4f}",
            "",
            "TRADING METRICS:",
            f"Total Trades: {metrics.get('total_trades', 0)}",
            f"Win Rate: {metrics.get('win_rate', 0):.2%}",
            f"Profit Factor: {metrics.get('profit_factor', 0):.4f}",
            f"Avg Win: {metrics.get('avg_win', 0):.4f}",
            f"Avg Loss: {metrics.get('avg_loss', 0):.4f}",
            "="*60
        ]
        
        return "\n".join(report)
