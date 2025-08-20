"""
Reinforcement Learning Portfolio Manager
Uses RL to optimize portfolio allocation across alpha models
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# RL imports
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    # Create dummy classes for when RL is not available
    class DummyEnv:
        def __init__(self):
            pass
        def reset(self, seed=None, options=None):
            return np.array([]), {}
        def step(self, action):
            return np.array([]), 0, True, False, {}
    
    class gym:
        Env = DummyEnv
    
    class spaces:
        @staticmethod
        def Box(low, high, shape, dtype):
            return None

class PortfolioEnvironment(gym.Env):
    """
    Custom gym environment for portfolio optimization
    """
    
    def __init__(self, returns_data: pd.DataFrame, alpha_signals: Dict[str, pd.DataFrame], 
                 transaction_cost: float = 0.001, max_position: float = 0.3):
        """
        Initialize portfolio environment
        
        Args:
            returns_data: Asset returns data
            alpha_signals: Alpha model signals
            transaction_cost: Transaction cost per trade
            max_position: Maximum position size per asset
        """
        super().__init__()
        
        self.returns_data = returns_data
        self.alpha_signals = alpha_signals
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Prepare combined features
        self._prepare_features()
        
        # Action space: portfolio weights for each asset/model combination
        self.n_assets = len(self.feature_matrix.columns)
        self.action_space = spaces.Box(
            low=-self.max_position,
            high=self.max_position,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Observation space: features + current portfolio + performance metrics
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets + 5,),  # features + portfolio + metrics
            dtype=np.float32
        )
        
        self.reset()
    
    def _prepare_features(self):
        """Prepare feature matrix from alpha signals"""
        feature_dfs = []
        
        for model_name, signals in self.alpha_signals.items():
            if not signals.empty:
                # Add model name prefix to avoid conflicts
                signals_renamed = signals.add_prefix(f'{model_name}_')
                feature_dfs.append(signals_renamed)
        
        if feature_dfs:
            self.feature_matrix = pd.concat(feature_dfs, axis=1, join='inner')
            self.feature_matrix = self.feature_matrix.fillna(0)
        else:
            # Fallback to returns data
            self.feature_matrix = self.returns_data.copy()
        
        # Normalize features
        self.feature_matrix = (self.feature_matrix - self.feature_matrix.mean()) / (
            self.feature_matrix.std() + 1e-8
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_portfolio = np.zeros(self.n_assets)
        self.portfolio_value = 1.0
        self.cumulative_return = 0.0
        self.max_drawdown = 0.0
        self.peak_value = 1.0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment"""
        # Ensure action is valid
        action = np.clip(action, -self.max_position, self.max_position)
        action = action / (np.sum(np.abs(action)) + 1e-8)  # Normalize to prevent extreme leverage
        
        # Calculate transaction costs
        portfolio_change = np.abs(action - self.current_portfolio)
        transaction_costs = np.sum(portfolio_change) * self.transaction_cost
        
        # Update portfolio
        self.current_portfolio = action
        
        # Calculate returns if we have data
        if self.current_step < len(self.feature_matrix) - 1:
            # Use feature matrix as proxy for returns (simplified)
            period_returns = self.feature_matrix.iloc[self.current_step].values
            
            # Portfolio return
            portfolio_return = np.dot(self.current_portfolio, period_returns) - transaction_costs
            
            # Update portfolio value
            self.portfolio_value *= (1 + portfolio_return)
            self.cumulative_return += portfolio_return
            
            # Update drawdown
            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value
            
            current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Calculate reward (risk-adjusted return)
            sharpe_reward = portfolio_return / (np.std(period_returns) + 1e-8)
            drawdown_penalty = -current_drawdown * 2
            
            reward = sharpe_reward + drawdown_penalty
        else:
            reward = 0
        
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.feature_matrix) - 1
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_step >= len(self.feature_matrix):
            features = np.zeros(self.n_assets)
        else:
            features = self.feature_matrix.iloc[self.current_step].values
        
        # Combine features with portfolio state and metrics
        observation = np.concatenate([
            features,
            [self.cumulative_return],
            [self.max_drawdown],
            [self.portfolio_value],
            [self.current_step / len(self.feature_matrix)],  # Progress
            [np.sum(np.abs(self.current_portfolio))]  # Portfolio concentration
        ])
        
        return observation.astype(np.float32)


class RLPortfolioManager:
    """
    Reinforcement Learning-based Portfolio Manager
    """
    
    def __init__(self, model_type: str = 'PPO'):
        """
        Initialize RL Portfolio Manager
        
        Args:
            model_type: Type of RL model ('PPO' or 'A2C')
        """
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        if not RL_AVAILABLE:
            self.logger.warning("stable-baselines3 not available. Using fallback portfolio optimization.")
    
    def optimize_portfolio(self, alpha_signals: Dict[str, pd.DataFrame], 
                          market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Optimize portfolio allocation using RL
        
        Args:
            alpha_signals: Alpha model signals
            market_data: Market data dictionary
            
        Returns:
            Portfolio weights DataFrame
        """
        self.logger.info("Optimizing portfolio with RL...")
        
        if not RL_AVAILABLE:
            return self._fallback_optimization(alpha_signals, market_data)
        
        # Prepare returns data
        returns_data = self._prepare_returns_data(market_data)
        
        if returns_data.empty or not alpha_signals:
            self.logger.warning("Insufficient data for RL optimization")
            return self._fallback_optimization(alpha_signals, market_data)
        
        try:
            # Create environment
            env = PortfolioEnvironment(returns_data, alpha_signals)
            
            # Train RL model
            portfolio_weights = self._train_and_predict(env, alpha_signals)
            
            self.logger.info(f"RL optimization completed. Portfolio shape: {portfolio_weights.shape}")
            return portfolio_weights
            
        except Exception as e:
            self.logger.error(f"RL optimization failed: {str(e)}")
            return self._fallback_optimization(alpha_signals, market_data)
    
    def _prepare_returns_data(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare returns data from market data"""
        returns_list = []
        
        for symbol, data in market_data.items():
            if 'Returns' in data.columns:
                returns_list.append(data['Returns'].rename(symbol))
        
        if returns_list:
            returns_df = pd.concat(returns_list, axis=1, join='inner')
            return returns_df.fillna(0)
        
        return pd.DataFrame()
    
    def _train_and_predict(self, env: PortfolioEnvironment, 
                          alpha_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Train RL model and generate predictions"""
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: env])
        
        # Initialize model
        if self.model_type == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                vec_env,
                verbose=0,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                seed=42
            )
        else:  # A2C
            self.model = A2C(
                'MlpPolicy',
                vec_env,
                verbose=0,
                learning_rate=0.0003,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.01,
                vf_coef=0.25,
                seed=42
            )
        
        # Train model
        self.logger.info(f"Training {self.model_type} model...")
        self.model.learn(total_timesteps=10000)
        self.is_trained = True
        
        # Generate portfolio weights
        portfolio_weights = self._generate_portfolio_weights(env)
        
        return portfolio_weights
    
    def _generate_portfolio_weights(self, env: PortfolioEnvironment) -> pd.DataFrame:
        """Generate portfolio weights using trained model"""
        
        weights_list = []
        obs, _ = env.reset()
        
        for i in range(len(env.feature_matrix)):
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Normalize weights
            action = action / (np.sum(np.abs(action)) + 1e-8)
            
            weights_list.append(action)
            
            obs, _, done, _, _ = env.step(action)
            
            if done:
                break
        
        # Create DataFrame
        if weights_list:
            weights_df = pd.DataFrame(
                weights_list,
                index=env.feature_matrix.index[:len(weights_list)],
                columns=[f'weight_{i}' for i in range(len(weights_list[0]))]
            )
            
            return weights_df
        
        return pd.DataFrame()
    
    def _fallback_optimization(self, alpha_signals: Dict[str, pd.DataFrame], 
                              market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Fallback portfolio optimization using simple signal-based approach
        """
        self.logger.info("Using fallback portfolio optimization...")
        
        # Combine all signals
        all_signals = []
        signal_names = []
        
        for model_name, signals in alpha_signals.items():
            if not signals.empty:
                # Take composite signal if available, otherwise mean of all signals
                if 'composite_momentum' in signals.columns:
                    all_signals.append(signals['composite_momentum'])
                    signal_names.append(f'{model_name}_composite')
                elif 'composite_mean_reversion' in signals.columns:
                    all_signals.append(signals['composite_mean_reversion'])
                    signal_names.append(f'{model_name}_composite')
                else:
                    # Use mean of all signals
                    signal_mean = signals.mean(axis=1)
                    all_signals.append(signal_mean)
                    signal_names.append(f'{model_name}_mean')
        
        if not all_signals:
            # Create dummy weights
            self.logger.warning("No signals available, creating dummy weights")
            dummy_index = next(iter(market_data.values())).index
            return pd.DataFrame(
                0.1,  # Equal small weights
                index=dummy_index,
                columns=['dummy_weight']
            )
        
        # Combine signals
        combined_signals = pd.concat(all_signals, axis=1, join='inner')
        combined_signals.columns = signal_names
        
        # Calculate signal-based weights
        portfolio_weights = self._calculate_signal_weights(combined_signals)
        
        return portfolio_weights
    
    def _calculate_signal_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio weights based on signals
        """
        # Normalize signals to weights
        weights = signals.copy()
        
        # Apply signal transformation
        weights = np.tanh(weights)  # Bound signals between -1 and 1
        
        # Ensure weights sum to reasonable levels (prevent extreme leverage)
        for idx in weights.index:
            row_sum = np.sum(np.abs(weights.loc[idx]))
            if row_sum > 1.0:
                weights.loc[idx] = weights.loc[idx] / row_sum
        
        # Add volatility-based adjustments if possible
        weights = self._apply_volatility_targeting(weights, signals)
        
        return weights
    
    def _apply_volatility_targeting(self, weights: pd.DataFrame, 
                                   signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply volatility targeting to portfolio weights
        """
        # Calculate signal volatilities
        signal_vols = signals.rolling(window=20).std()
        
        # Inverse volatility weighting
        vol_weights = 1 / (signal_vols + 1e-8)
        vol_weights = vol_weights.div(vol_weights.sum(axis=1), axis=0)
        
        # Combine with signal weights
        adjusted_weights = weights * vol_weights
        
        # Normalize again
        for idx in adjusted_weights.index:
            row_sum = np.sum(np.abs(adjusted_weights.loc[idx]))
            if row_sum > 0:
                adjusted_weights.loc[idx] = adjusted_weights.loc[idx] / row_sum * 0.8  # 80% max allocation
        
        return adjusted_weights
    
    def evaluate_portfolio_performance(self, weights: pd.DataFrame, 
                                     returns: pd.DataFrame) -> Dict:
        """
        Evaluate portfolio performance metrics
        
        Args:
            weights: Portfolio weights DataFrame
            returns: Asset returns DataFrame
            
        Returns:
            Performance metrics dictionary
        """
        if weights.empty or returns.empty:
            return {}
        
        # Align weights and returns
        common_index = weights.index.intersection(returns.index)
        
        if len(common_index) == 0:
            return {}
        
        weights_aligned = weights.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        # Calculate portfolio returns
        # For simplicity, assume weights apply to first asset
        if len(returns_aligned.columns) > 0:
            first_asset_returns = returns_aligned.iloc[:, 0]
            
            # Simple portfolio return calculation
            portfolio_returns = weights_aligned.mean(axis=1) * first_asset_returns
        else:
            portfolio_returns = pd.Series(0, index=common_index)
        
        # Calculate metrics
        metrics = {
            'total_return': portfolio_returns.sum(),
            'volatility': portfolio_returns.std(),
            'sharpe_ratio': portfolio_returns.mean() / (portfolio_returns.std() + 1e-8) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'hit_rate': (portfolio_returns > 0).sum() / len(portfolio_returns),
            'avg_weight': weights_aligned.mean().mean()
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
