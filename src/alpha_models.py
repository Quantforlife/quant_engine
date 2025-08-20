"""
Alpha Models Module
Implements momentum and mean reversion strategies for different market regimes
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AlphaModels:
    """
    Generates alpha signals using momentum and mean reversion strategies
    """
    
    def __init__(self):
        """Initialize alpha models"""
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
        # Model parameters
        self.momentum_params = {
            'short_window': 10,
            'long_window': 30,
            'breakout_window': 20,
            'volume_threshold': 1.2
        }
        
        self.mean_reversion_params = {
            'bb_window': 20,
            'bb_std': 2,
            'rsi_window': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'zscore_window': 20,
            'zscore_threshold': 2.0
        }
        
        # Regime assignments
        self.regime_model_assignment = {
            0: 'momentum',      # Bullish -> Momentum
            1: 'mean_reversion', # Bearish -> Mean Reversion
            2: 'momentum'       # Volatile -> Momentum
        }
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame], 
                        regime_data: Dict) -> Dict[str, pd.DataFrame]:
        """
        Generate alpha signals for all models and regimes
        
        Args:
            market_data: Market data dictionary
            regime_data: Regime detection results
            
        Returns:
            Dictionary of alpha signals by model
        """
        self.logger.info("Generating alpha model signals...")
        
        signals = {}
        
        # Generate momentum signals
        momentum_signals = self._generate_momentum_signals(market_data)
        if not momentum_signals.empty:
            signals['momentum'] = momentum_signals
        
        # Generate mean reversion signals
        mean_rev_signals = self._generate_mean_reversion_signals(market_data)
        if not mean_rev_signals.empty:
            signals['mean_reversion'] = mean_rev_signals
        
        # Combine signals with regime information
        if regime_data and 'regime_labels' in regime_data:
            regime_adjusted_signals = self._adjust_signals_for_regimes(
                signals, regime_data, market_data
            )
            signals.update(regime_adjusted_signals)
        
        self.logger.info(f"Generated signals for {len(signals)} models")
        return signals
    
    def _generate_momentum_signals(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate momentum-based alpha signals
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Momentum signals DataFrame
        """
        self.logger.info("Generating momentum signals...")
        
        momentum_signals = []
        
        for symbol, data in market_data.items():
            if data.empty:
                continue
            
            symbol_signals = pd.DataFrame(index=data.index)
            
            # Moving average crossover
            if all(col in data.columns for col in ['MA_20', 'MA_50']):
                ma_cross = np.where(data['MA_20'] > data['MA_50'], 1, -1)
                symbol_signals[f'{symbol}_ma_cross'] = ma_cross
            
            # Price momentum
            if 'Close' in data.columns:
                price_momentum = data['Close'].pct_change(periods=self.momentum_params['short_window'])
                symbol_signals[f'{symbol}_price_momentum'] = np.tanh(price_momentum * 100)  # Scale and bound
            
            # Breakout signals
            if 'Close' in data.columns:
                rolling_max = data['Close'].rolling(
                    window=self.momentum_params['breakout_window']
                ).max()
                rolling_min = data['Close'].rolling(
                    window=self.momentum_params['breakout_window']
                ).min()
                
                breakout_up = (data['Close'] > rolling_max.shift(1)).astype(int)
                breakout_down = (data['Close'] < rolling_min.shift(1)).astype(int)
                
                symbol_signals[f'{symbol}_breakout'] = breakout_up - breakout_down
            
            # Volume confirmation
            if all(col in data.columns for col in ['Volume', 'Close']):
                avg_volume = data['Volume'].rolling(window=20).mean()
                volume_spike = (data['Volume'] > avg_volume * self.momentum_params['volume_threshold']).astype(int)
                
                # Combine with price movement
                price_change = np.sign(data['Close'].pct_change())
                volume_momentum = volume_spike * price_change
                symbol_signals[f'{symbol}_volume_momentum'] = volume_momentum
            
            # RSI momentum
            if 'RSI' in data.columns:
                rsi_momentum = np.where(
                    data['RSI'] > 50,
                    (data['RSI'] - 50) / 50,  # Bullish momentum
                    (data['RSI'] - 50) / 50   # Bearish momentum
                )
                symbol_signals[f'{symbol}_rsi_momentum'] = rsi_momentum
            
            momentum_signals.append(symbol_signals)
        
        if momentum_signals:
            combined_signals = pd.concat(momentum_signals, axis=1, join='inner')
            
            # Create composite momentum signal
            if not combined_signals.empty:
                combined_signals['composite_momentum'] = combined_signals.mean(axis=1)
                
                self.logger.info(f"Generated momentum signals with shape: {combined_signals.shape}")
                return combined_signals.dropna()
        
        return pd.DataFrame()
    
    def _generate_mean_reversion_signals(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate mean reversion alpha signals
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Mean reversion signals DataFrame
        """
        self.logger.info("Generating mean reversion signals...")
        
        mean_rev_signals = []
        
        for symbol, data in market_data.items():
            if data.empty:
                continue
            
            symbol_signals = pd.DataFrame(index=data.index)
            
            # Bollinger Bands mean reversion
            if all(col in data.columns for col in ['Close', 'BB_Upper', 'BB_Lower', 'BB_Middle']):
                bb_position = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
                
                # Signal when price is extreme relative to bands
                bb_signal = np.where(
                    bb_position > 0.8, -1,  # Overbought - sell signal
                    np.where(bb_position < 0.2, 1, 0)  # Oversold - buy signal
                )
                symbol_signals[f'{symbol}_bb_reversion'] = bb_signal
                
                # Mean reversion strength
                distance_from_mean = (data['Close'] - data['BB_Middle']) / data['BB_Middle']
                reversion_strength = -np.tanh(distance_from_mean * 5)  # Invert for mean reversion
                symbol_signals[f'{symbol}_bb_strength'] = reversion_strength
            
            # RSI mean reversion
            if 'RSI' in data.columns:
                rsi_signal = np.where(
                    data['RSI'] > self.mean_reversion_params['rsi_overbought'], -1,
                    np.where(data['RSI'] < self.mean_reversion_params['rsi_oversold'], 1, 0)
                )
                symbol_signals[f'{symbol}_rsi_reversion'] = rsi_signal
            
            # Z-score mean reversion
            if 'Returns' in data.columns:
                returns_zscore = (
                    data['Returns'] - data['Returns'].rolling(
                        window=self.mean_reversion_params['zscore_window']
                    ).mean()
                ) / data['Returns'].rolling(
                    window=self.mean_reversion_params['zscore_window']
                ).std()
                
                zscore_signal = np.where(
                    returns_zscore > self.mean_reversion_params['zscore_threshold'], -1,
                    np.where(returns_zscore < -self.mean_reversion_params['zscore_threshold'], 1, 0)
                )
                symbol_signals[f'{symbol}_zscore_reversion'] = zscore_signal
            
            # Price vs moving average reversion
            if all(col in data.columns for col in ['Close', 'MA_20']):
                price_ma_ratio = data['Close'] / data['MA_20'] - 1
                ma_reversion = -np.tanh(price_ma_ratio * 10)  # Invert for mean reversion
                symbol_signals[f'{symbol}_ma_reversion'] = ma_reversion
            
            # Volatility-adjusted signals
            if 'Volatility' in data.columns:
                # Scale signals by inverse volatility (higher vol = smaller signals)
                vol_adj_factor = 1 / (1 + data['Volatility'] * 10)
                
                for col in symbol_signals.columns:
                    if col != f'{symbol}_volatility_factor':
                        symbol_signals[col] = symbol_signals[col] * vol_adj_factor
                
                symbol_signals[f'{symbol}_volatility_factor'] = vol_adj_factor
            
            mean_rev_signals.append(symbol_signals)
        
        if mean_rev_signals:
            combined_signals = pd.concat(mean_rev_signals, axis=1, join='inner')
            
            # Create composite mean reversion signal
            if not combined_signals.empty:
                reversion_cols = [col for col in combined_signals.columns if 'reversion' in col]
                if reversion_cols:
                    combined_signals['composite_mean_reversion'] = combined_signals[reversion_cols].mean(axis=1)
                
                self.logger.info(f"Generated mean reversion signals with shape: {combined_signals.shape}")
                return combined_signals.dropna()
        
        return pd.DataFrame()
    
    def _adjust_signals_for_regimes(self, signals: Dict[str, pd.DataFrame], 
                                   regime_data: Dict, 
                                   market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Adjust alpha signals based on detected market regimes
        
        Args:
            signals: Dictionary of alpha signals
            regime_data: Regime detection results
            market_data: Market data dictionary
            
        Returns:
            Regime-adjusted signals dictionary
        """
        self.logger.info("Adjusting signals for market regimes...")
        
        adjusted_signals = {}
        
        if 'regime_labels' not in regime_data:
            return adjusted_signals
        
        # Get regime information
        regime_labels = regime_data['regime_labels']
        regime_probs = regime_data.get('regime_probs', pd.DataFrame())
        
        # Create regime-based signal adjustments
        for model_name, model_signals in signals.items():
            if model_signals.empty:
                continue
            
            # Align regime data with signal data
            common_index = model_signals.index.intersection(
                pd.Index(regime_data['feature_data'].index[:len(regime_labels)])
            )
            
            if len(common_index) == 0:
                continue
            
            # Create regime-adjusted signals
            regime_adjusted = model_signals.loc[common_index].copy()
            aligned_regimes = regime_labels[:len(common_index)]
            
            # Adjust signal strength based on regime assignment
            for i, (idx, regime) in enumerate(zip(common_index, aligned_regimes)):
                preferred_model = self.regime_model_assignment.get(regime, model_name)
                
                if preferred_model == model_name:
                    # Boost signals when model matches regime
                    regime_adjusted.loc[idx] = regime_adjusted.loc[idx] * 1.2
                else:
                    # Reduce signals when model doesn't match regime
                    regime_adjusted.loc[idx] = regime_adjusted.loc[idx] * 0.6
            
            # Add regime confidence weighting
            if not regime_probs.empty:
                regime_probs_aligned = regime_probs.loc[common_index]
                
                # Calculate regime confidence (max probability)
                regime_confidence = regime_probs_aligned.max(axis=1)
                
                # Apply confidence weighting
                for col in regime_adjusted.columns:
                    regime_adjusted[col] = regime_adjusted[col] * regime_confidence
            
            adjusted_signals[f'{model_name}_regime_adjusted'] = regime_adjusted
        
        # Create regime-specific combined signals
        self._create_regime_specific_signals(adjusted_signals, signals, regime_data)
        
        return adjusted_signals
    
    def _create_regime_specific_signals(self, adjusted_signals: Dict[str, pd.DataFrame],
                                       original_signals: Dict[str, pd.DataFrame],
                                       regime_data: Dict):
        """
        Create regime-specific combined signals
        
        Args:
            adjusted_signals: Dictionary to store regime-adjusted signals
            original_signals: Original alpha signals
            regime_data: Regime detection results
        """
        if 'regime_labels' not in regime_data:
            return
        
        regime_labels = regime_data['regime_labels']
        
        # For each regime, create optimal signal combination
        for regime_id in range(3):  # Assuming 3 regimes
            regime_name = {0: 'bullish', 1: 'bearish', 2: 'volatile'}[regime_id]
            preferred_model = self.regime_model_assignment[regime_id]
            
            if preferred_model in original_signals:
                preferred_signals = original_signals[preferred_model].copy()
                
                # Create regime mask
                regime_mask = np.array(regime_labels) == regime_id
                
                if len(regime_mask) > len(preferred_signals):
                    regime_mask = regime_mask[:len(preferred_signals)]
                elif len(regime_mask) < len(preferred_signals):
                    # Extend mask with False values
                    extended_mask = np.zeros(len(preferred_signals), dtype=bool)
                    extended_mask[:len(regime_mask)] = regime_mask
                    regime_mask = extended_mask
                
                # Apply regime-specific adjustments
                regime_specific_signals = preferred_signals.copy()
                
                # Zero out signals when not in preferred regime
                for col in regime_specific_signals.columns:
                    regime_specific_signals.loc[~regime_mask, col] = 0
                
                adjusted_signals[f'{regime_name}_regime_signals'] = regime_specific_signals
    
    def calculate_signal_quality(self, signals: Dict[str, pd.DataFrame], 
                                market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate quality metrics for generated signals
        
        Args:
            signals: Alpha signals dictionary
            market_data: Market data dictionary
            
        Returns:
            Signal quality metrics dictionary
        """
        quality_metrics = {}
        
        for signal_name, signal_df in signals.items():
            if signal_df.empty:
                continue
            
            metrics = {}
            
            # Signal statistics
            metrics['signal_count'] = len(signal_df)
            metrics['non_zero_signals'] = (signal_df != 0).sum().sum()
            metrics['signal_coverage'] = metrics['non_zero_signals'] / (len(signal_df) * len(signal_df.columns))
            
            # Signal strength distribution
            signal_values = signal_df.values.flatten()
            signal_values = signal_values[~np.isnan(signal_values)]
            
            metrics['mean_signal_strength'] = np.mean(np.abs(signal_values))
            metrics['signal_volatility'] = np.std(signal_values)
            metrics['signal_skewness'] = pd.Series(signal_values).skew()
            
            # Turnover (signal changes)
            turnover_rates = []
            for col in signal_df.columns:
                signal_changes = (signal_df[col].diff() != 0).sum()
                turnover_rate = signal_changes / len(signal_df)
                turnover_rates.append(turnover_rate)
            
            metrics['avg_turnover_rate'] = np.mean(turnover_rates)
            
            quality_metrics[signal_name] = metrics
        
        return quality_metrics
