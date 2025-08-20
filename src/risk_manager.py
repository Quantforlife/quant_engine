"""
Risk Management Module
Implements Kelly Criterion position sizing and volatility targeting
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    """
    Manages portfolio risk through position sizing and volatility controls
    """
    
    def __init__(self, target_volatility: float = 0.15, max_position: float = 0.25,
                 max_leverage: float = 1.0, kelly_lookback: int = 252):
        """
        Initialize Risk Manager
        
        Args:
            target_volatility: Target portfolio volatility (annualized)
            max_position: Maximum position size per asset
            max_leverage: Maximum total leverage
            kelly_lookback: Lookback period for Kelly Criterion calculation
        """
        self.logger = logging.getLogger(__name__)
        self.target_volatility = target_volatility
        self.max_position = max_position
        self.max_leverage = max_leverage
        self.kelly_lookback = kelly_lookback
        
        # Risk metrics tracking
        self.risk_metrics = {}
        
    def apply_risk_controls(self, portfolio_weights: pd.DataFrame, 
                           market_data: Dict[str, pd.DataFrame],
                           alpha_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Apply comprehensive risk controls to portfolio weights
        
        Args:
            portfolio_weights: Raw portfolio weights from RL manager
            market_data: Market data dictionary
            alpha_signals: Alpha model signals
            
        Returns:
            Risk-adjusted portfolio weights
        """
        self.logger.info("Applying risk management controls...")
        
        if portfolio_weights.empty:
            self.logger.warning("Empty portfolio weights provided")
            return portfolio_weights
        
        # Step 1: Calculate returns data for risk calculations
        returns_data = self._prepare_returns_data(market_data)
        
        # Step 2: Apply Kelly Criterion position sizing
        kelly_adjusted_weights = self._apply_kelly_criterion(
            portfolio_weights, returns_data, alpha_signals
        )
        
        # Step 3: Apply volatility targeting
        vol_adjusted_weights = self._apply_volatility_targeting(
            kelly_adjusted_weights, returns_data
        )
        
        # Step 4: Apply position limits
        position_limited_weights = self._apply_position_limits(vol_adjusted_weights)
        
        # Step 5: Apply leverage constraints
        final_weights = self._apply_leverage_constraints(position_limited_weights)
        
        # Step 6: Calculate and store risk metrics
        self._calculate_risk_metrics(final_weights, returns_data)
        
        self.logger.info(f"Risk controls applied. Final weights shape: {final_weights.shape}")
        return final_weights
    
    def _prepare_returns_data(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare returns data for risk calculations
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Returns DataFrame
        """
        returns_list = []
        
        for symbol, data in market_data.items():
            if 'Returns' in data.columns and not data['Returns'].empty:
                returns_list.append(data['Returns'].rename(symbol))
        
        if returns_list:
            returns_df = pd.concat(returns_list, axis=1, join='inner')
            return returns_df.fillna(0)
        else:
            self.logger.warning("No returns data available for risk calculations")
            return pd.DataFrame()
    
    def _apply_kelly_criterion(self, weights: pd.DataFrame, 
                              returns_data: pd.DataFrame,
                              alpha_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Apply Kelly Criterion for optimal position sizing
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns
            alpha_signals: Alpha model signals
            
        Returns:
            Kelly-adjusted weights
        """
        self.logger.info("Applying Kelly Criterion position sizing...")
        
        if returns_data.empty or weights.empty:
            return weights
        
        kelly_weights = weights.copy()
        
        # Align data
        common_index = weights.index.intersection(returns_data.index)
        
        if len(common_index) < self.kelly_lookback:
            self.logger.warning(f"Insufficient data for Kelly calculation. Need {self.kelly_lookback}, got {len(common_index)}")
            return weights
        
        # Calculate Kelly fractions for each weight column
        for col in weights.columns:
            for i, date in enumerate(common_index[self.kelly_lookback:], self.kelly_lookback):
                # Get lookback window
                end_idx = common_index.get_loc(date)
                start_idx = max(0, end_idx - self.kelly_lookback)
                lookback_dates = common_index[start_idx:end_idx]
                
                if len(lookback_dates) < 20:  # Minimum data requirement
                    continue
                
                # Calculate Kelly fraction for each asset
                kelly_fractions = []
                
                for asset in returns_data.columns:
                    asset_returns = returns_data.loc[lookback_dates, asset]
                    
                    if len(asset_returns) == 0 or asset_returns.std() == 0:
                        kelly_fractions.append(0)
                        continue
                    
                    # Kelly formula: f = (bp - q) / b
                    # where b = odds, p = win probability, q = loss probability
                    
                    # Estimate win probability and average win/loss
                    wins = asset_returns[asset_returns > 0]
                    losses = asset_returns[asset_returns < 0]
                    
                    if len(wins) == 0 or len(losses) == 0:
                        kelly_fractions.append(0)
                        continue
                    
                    win_prob = len(wins) / len(asset_returns)
                    avg_win = wins.mean()
                    avg_loss = abs(losses.mean())
                    
                    if avg_loss == 0:
                        kelly_fractions.append(0)
                        continue
                    
                    # Kelly fraction
                    odds_ratio = avg_win / avg_loss
                    kelly_fraction = (odds_ratio * win_prob - (1 - win_prob)) / odds_ratio
                    
                    # Cap Kelly fraction to prevent extreme positions
                    kelly_fraction = np.clip(kelly_fraction, -0.25, 0.25)
                    kelly_fractions.append(kelly_fraction)
                
                # Apply Kelly scaling to weights
                if kelly_fractions and not all(f == 0 for f in kelly_fractions):
                    avg_kelly = np.mean([f for f in kelly_fractions if f != 0])
                    kelly_scaling = min(abs(avg_kelly), 1.0)  # Cap at 100%
                    
                    # Scale the weight by Kelly fraction
                    original_weight = weights.loc[date, col]
                    kelly_weights.loc[date, col] = original_weight * kelly_scaling
        
        return kelly_weights
    
    def _apply_volatility_targeting(self, weights: pd.DataFrame, 
                                   returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply volatility targeting to maintain consistent risk levels
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns
            
        Returns:
            Volatility-adjusted weights
        """
        self.logger.info("Applying volatility targeting...")
        
        if returns_data.empty or weights.empty:
            return weights
        
        vol_adjusted_weights = weights.copy()
        vol_window = min(60, len(returns_data))  # 60-day or available data
        
        # Calculate realized portfolio volatility
        for date in weights.index:
            if date not in returns_data.index:
                continue
            
            # Get historical window
            date_idx = returns_data.index.get_loc(date)
            start_idx = max(0, date_idx - vol_window)
            hist_returns = returns_data.iloc[start_idx:date_idx]
            
            if len(hist_returns) < 20:  # Minimum data requirement
                continue
            
            # Calculate portfolio volatility
            if len(returns_data.columns) > 0:
                # Use first asset as proxy for portfolio volatility
                asset_vol = hist_returns.iloc[:, 0].std() * np.sqrt(252)  # Annualized
                
                if asset_vol > 0:
                    # Calculate volatility scaling factor
                    vol_scaling = self.target_volatility / asset_vol
                    vol_scaling = np.clip(vol_scaling, 0.1, 3.0)  # Reasonable bounds
                    
                    # Apply scaling to all weights for this date
                    vol_adjusted_weights.loc[date] = weights.loc[date] * vol_scaling
        
        return vol_adjusted_weights
    
    def _apply_position_limits(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply position size limits to prevent concentration risk
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Position-limited weights
        """
        self.logger.info("Applying position limits...")
        
        limited_weights = weights.copy()
        
        # Apply individual position limits
        limited_weights = limited_weights.clip(-self.max_position, self.max_position)
        
        # Apply per-row (date) position limits
        for date in limited_weights.index:
            row_weights = limited_weights.loc[date]
            
            # Check if any position exceeds limit
            max_abs_weight = row_weights.abs().max()
            
            if max_abs_weight > self.max_position:
                # Scale down proportionally
                scaling_factor = self.max_position / max_abs_weight
                limited_weights.loc[date] = row_weights * scaling_factor
        
        return limited_weights
    
    def _apply_leverage_constraints(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply leverage constraints to limit total exposure
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Leverage-constrained weights
        """
        self.logger.info("Applying leverage constraints...")
        
        constrained_weights = weights.copy()
        
        for date in constrained_weights.index:
            row_weights = constrained_weights.loc[date]
            total_leverage = row_weights.abs().sum()
            
            if total_leverage > self.max_leverage:
                # Scale down to meet leverage constraint
                scaling_factor = self.max_leverage / total_leverage
                constrained_weights.loc[date] = row_weights * scaling_factor
        
        return constrained_weights
    
    def _calculate_risk_metrics(self, weights: pd.DataFrame, 
                               returns_data: pd.DataFrame):
        """
        Calculate and store risk metrics for monitoring
        
        Args:
            weights: Final portfolio weights
            returns_data: Historical returns
        """
        if weights.empty or returns_data.empty:
            return
        
        # Calculate portfolio-level metrics
        self.risk_metrics = {
            'avg_leverage': weights.abs().sum(axis=1).mean(),
            'max_leverage': weights.abs().sum(axis=1).max(),
            'avg_concentration': weights.abs().max(axis=1).mean(),
            'max_concentration': weights.abs().max(axis=1).max(),
            'weight_volatility': weights.std().mean(),
            'turnover': self._calculate_turnover(weights)
        }
        
        # Calculate realized volatility if possible
        if len(returns_data.columns) > 0:
            # Simple portfolio return approximation
            portfolio_returns = weights.mean(axis=1) * returns_data.iloc[:, 0]
            common_dates = portfolio_returns.dropna()
            
            if len(common_dates) > 20:
                realized_vol = common_dates.std() * np.sqrt(252)
                self.risk_metrics['realized_volatility'] = realized_vol
                self.risk_metrics['vol_target_ratio'] = realized_vol / self.target_volatility
        
        self.logger.info(f"Risk metrics calculated: {self.risk_metrics}")
    
    def _calculate_turnover(self, weights: pd.DataFrame) -> float:
        """
        Calculate portfolio turnover rate
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Average turnover rate
        """
        if len(weights) < 2:
            return 0.0
        
        # Calculate weight changes
        weight_changes = weights.diff().abs()
        
        # Turnover is sum of absolute weight changes
        daily_turnover = weight_changes.sum(axis=1)
        
        return daily_turnover.mean()
    
    def get_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report
        
        Returns:
            Risk report dictionary
        """
        report = {
            'risk_parameters': {
                'target_volatility': self.target_volatility,
                'max_position': self.max_position,
                'max_leverage': self.max_leverage,
                'kelly_lookback': self.kelly_lookback
            },
            'risk_metrics': self.risk_metrics.copy(),
            'risk_warnings': []
        }
        
        # Add risk warnings
        if 'max_leverage' in self.risk_metrics:
            if self.risk_metrics['max_leverage'] > self.max_leverage * 0.9:
                report['risk_warnings'].append("High leverage detected")
        
        if 'max_concentration' in self.risk_metrics:
            if self.risk_metrics['max_concentration'] > self.max_position * 0.9:
                report['risk_warnings'].append("High concentration risk")
        
        if 'vol_target_ratio' in self.risk_metrics:
            if abs(self.risk_metrics['vol_target_ratio'] - 1.0) > 0.5:
                report['risk_warnings'].append("Volatility significantly off target")
        
        return report
    
    def calculate_var(self, weights: pd.DataFrame, returns_data: pd.DataFrame, 
                     confidence_level: float = 0.05) -> Dict:
        """
        Calculate Value at Risk (VaR) metrics
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns
            confidence_level: VaR confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR metrics dictionary
        """
        if weights.empty or returns_data.empty:
            return {}
        
        # Align data
        common_index = weights.index.intersection(returns_data.index)
        
        if len(common_index) < 30:
            return {}
        
        # Calculate portfolio returns (simplified)
        if len(returns_data.columns) > 0:
            portfolio_returns = weights.loc[common_index].mean(axis=1) * returns_data.loc[common_index].iloc[:, 0]
        else:
            return {}
        
        portfolio_returns = portfolio_returns.dropna()
        
        if len(portfolio_returns) < 30:
            return {}
        
        # Calculate VaR metrics
        var_metrics = {
            'daily_var': np.percentile(portfolio_returns, confidence_level * 100),
            'daily_cvar': portfolio_returns[portfolio_returns <= np.percentile(
                portfolio_returns, confidence_level * 100
            )].mean(),
            'monthly_var': np.percentile(portfolio_returns, confidence_level * 100) * np.sqrt(21),
            'annual_var': np.percentile(portfolio_returns, confidence_level * 100) * np.sqrt(252)
        }
        
        return var_metrics
