"""
Regime Detection Module
Implements Hidden Markov Models for market regime identification
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RegimeDetector:
    """
    Detects market regimes using Hidden Markov Models
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector
        
        Args:
            n_regimes: Number of market regimes to detect (default: 3)
        """
        self.logger = logging.getLogger(__name__)
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler()
        self.regime_names = {0: 'Bullish', 1: 'Bearish', 2: 'Volatile'}
        
    def detect_regimes(self, market_data: Dict[str, pd.DataFrame], 
                      economic_data: pd.DataFrame) -> Dict:
        """
        Detect market regimes using HMM
        
        Args:
            market_data: Dictionary of market data
            economic_data: Economic indicators DataFrame
            
        Returns:
            Dictionary containing regime information
        """
        self.logger.info("Starting regime detection...")
        
        # Prepare feature matrix
        features_df = self._prepare_features(market_data, economic_data)
        
        if features_df.empty:
            self.logger.error("No valid features for regime detection")
            return {}
        
        # Fit HMM model
        self.logger.info(f"Fitting HMM with {self.n_regimes} regimes...")
        regime_labels, regime_probs = self._fit_hmm(features_df)
        
        # Analyze regimes
        regime_analysis = self._analyze_regimes(features_df, regime_labels)
        
        result = {
            'regime_labels': regime_labels,
            'regime_probs': regime_probs,
            'regime_analysis': regime_analysis,
            'feature_data': features_df,
            'regime_transitions': self._calculate_transitions(regime_labels)
        }
        
        self.logger.info("Regime detection completed successfully")
        return result
    
    def _prepare_features(self, market_data: Dict[str, pd.DataFrame], 
                         economic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature matrix for HMM
        
        Args:
            market_data: Market data dictionary
            economic_data: Economic data DataFrame
            
        Returns:
            Features DataFrame
        """
        self.logger.info("Preparing features for regime detection...")
        
        features_list = []
        
        # Market features
        for symbol, data in market_data.items():
            if data.empty:
                continue
                
            # Return-based features
            features_list.append(data['Returns'].rename(f'{symbol}_returns'))
            features_list.append(data['Volatility'].rename(f'{symbol}_volatility'))
            
            # Technical indicators
            if 'RSI' in data.columns:
                features_list.append(data['RSI'].rename(f'{symbol}_rsi'))
            
            # Bollinger Band position
            if all(col in data.columns for col in ['Close', 'BB_Upper', 'BB_Lower']):
                bb_position = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
                features_list.append(bb_position.rename(f'{symbol}_bb_position'))
            
            # Moving average ratios
            if all(col in data.columns for col in ['Close', 'MA_20', 'MA_50']):
                ma_ratio_20 = data['Close'] / data['MA_20'] - 1
                ma_ratio_50 = data['Close'] / data['MA_50'] - 1
                features_list.append(ma_ratio_20.rename(f'{symbol}_ma20_ratio'))
                features_list.append(ma_ratio_50.rename(f'{symbol}_ma50_ratio'))
        
        # Economic features (resample to daily frequency)
        if not economic_data.empty:
            # Resample economic data to daily frequency
            econ_daily = economic_data.resample('D').ffill()
            
            for col in economic_data.columns:
                if '_change' in col:  # Use change variables
                    features_list.append(econ_daily[col].rename(f'econ_{col}'))
        
        # Combine all features
        if features_list:
            features_df = pd.concat(features_list, axis=1, join='inner')
            features_df = features_df.dropna()
            
            self.logger.info(f"Prepared {features_df.shape[1]} features with {features_df.shape[0]} observations")
            return features_df
        else:
            self.logger.error("No valid features created")
            return pd.DataFrame()
    
    def _fit_hmm(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Fit Hidden Markov Model
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Tuple of (regime_labels, regime_probabilities)
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df.values)
        
        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        try:
            self.model.fit(features_scaled)
            
            # Predict regimes
            regime_labels = self.model.predict(features_scaled)
            regime_probs = self.model.predict_proba(features_scaled)
            
            # Create regime probabilities DataFrame
            regime_probs_df = pd.DataFrame(
                regime_probs,
                index=features_df.index,
                columns=[f'Regime_{i}' for i in range(self.n_regimes)]
            )
            
            self.logger.info(f"HMM fitted successfully. Log-likelihood: {self.model.score(features_scaled):.2f}")
            
            return regime_labels, regime_probs_df
            
        except Exception as e:
            self.logger.error(f"Error fitting HMM: {str(e)}")
            # Fallback to simple volatility-based regimes
            return self._fallback_regime_detection(features_df)
    
    def _fallback_regime_detection(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Fallback regime detection using volatility quantiles
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Tuple of (regime_labels, regime_probabilities)
        """
        self.logger.warning("Using fallback regime detection based on volatility")
        
        # Use volatility of first market feature as proxy
        vol_cols = [col for col in features_df.columns if 'volatility' in col.lower()]
        
        if vol_cols:
            volatility = features_df[vol_cols[0]]
        else:
            # Calculate volatility from returns
            return_cols = [col for col in features_df.columns if 'returns' in col.lower()]
            if return_cols:
                volatility = features_df[return_cols[0]].rolling(window=20).std()
            else:
                # Use first column as fallback
                volatility = features_df.iloc[:, 0].rolling(window=20).std()
        
        # Define regimes based on volatility quantiles
        vol_q33 = volatility.quantile(0.33)
        vol_q67 = volatility.quantile(0.67)
        
        regime_labels = np.where(volatility <= vol_q33, 0,  # Low vol = Bullish
                                np.where(volatility <= vol_q67, 1, 2))  # High vol = Volatile, Medium = Bearish
        
        # Create dummy probabilities
        regime_probs = np.zeros((len(regime_labels), self.n_regimes))
        for i, label in enumerate(regime_labels):
            regime_probs[i, label] = 1.0
        
        regime_probs_df = pd.DataFrame(
            regime_probs,
            index=features_df.index,
            columns=[f'Regime_{i}' for i in range(self.n_regimes)]
        )
        
        return regime_labels, regime_probs_df
    
    def _analyze_regimes(self, features_df: pd.DataFrame, 
                        regime_labels: np.ndarray) -> Dict:
        """
        Analyze characteristics of detected regimes
        
        Args:
            features_df: Features DataFrame
            regime_labels: Regime labels array
            
        Returns:
            Regime analysis dictionary
        """
        analysis = {}
        
        for regime in range(self.n_regimes):
            regime_mask = regime_labels == regime
            regime_data = features_df[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            # Calculate regime statistics
            regime_stats = {
                'count': len(regime_data),
                'percentage': len(regime_data) / len(features_df) * 100,
                'mean_features': regime_data.mean().to_dict(),
                'std_features': regime_data.std().to_dict()
            }
            
            # Calculate average duration
            regime_series = pd.Series(regime_labels == regime, index=features_df.index)
            durations = []
            current_duration = 0
            
            for in_regime in regime_series:
                if in_regime:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            
            if current_duration > 0:
                durations.append(current_duration)
            
            regime_stats['avg_duration'] = np.mean(durations) if durations else 0
            regime_stats['median_duration'] = np.median(durations) if durations else 0
            
            analysis[f'regime_{regime}'] = regime_stats
        
        return analysis
    
    def _calculate_transitions(self, regime_labels: np.ndarray) -> pd.DataFrame:
        """
        Calculate regime transition matrix
        
        Args:
            regime_labels: Regime labels array
            
        Returns:
            Transition matrix DataFrame
        """
        transitions = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(regime_labels) - 1):
            current_regime = regime_labels[i]
            next_regime = regime_labels[i + 1]
            transitions[current_regime, next_regime] += 1
        
        # Normalize to probabilities
        row_sums = transitions.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transitions = transitions / row_sums[:, np.newaxis]
        
        # Create DataFrame
        transition_df = pd.DataFrame(
            transitions,
            index=[f'From_Regime_{i}' for i in range(self.n_regimes)],
            columns=[f'To_Regime_{i}' for i in range(self.n_regimes)]
        )
        
        return transition_df
    
    def predict_next_regime(self, current_features: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict next regime given current features
        
        Args:
            current_features: Current feature vector
            
        Returns:
            Tuple of (most_likely_regime, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call detect_regimes first.")
        
        # Scale features
        features_scaled = self.scaler.transform(current_features.reshape(1, -1))
        
        # Predict probabilities
        probs = self.model.predict_proba(features_scaled)
        most_likely_regime = np.argmax(probs[0])
        
        return most_likely_regime, probs[0]
