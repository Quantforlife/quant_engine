"""
Configuration Module
Central configuration for the Multi-Modal Regime-Switching Alpha Engine
"""

import os
from typing import List, Dict, Any
from datetime import timedelta

class Config:
    """
    Central configuration class for the entire platform
    """
    
    def __init__(self):
        """Initialize configuration with default values"""
        
        # Data Configuration
        self.MARKET_SYMBOLS = ['^GSPC', '^IXIC']  # S&P 500 and NASDAQ
        self.ECONOMIC_INDICATORS = [
            'CPIAUCSL',    # Consumer Price Index
            'GDP',         # Gross Domestic Product
            'UNRATE',      # Unemployment Rate
            'DGS10'        # 10-Year Treasury Constant Maturity Rate
        ]
        
        # Time Configuration
        self.LOOKBACK_DAYS = 3650  # 10+ years of data
        self.MIN_DATA_POINTS = 252  # Minimum trading days required
        
        # Regime Detection Configuration
        self.REGIME_CONFIG = {
            'n_regimes': 3,
            'hmm_iterations': 100,
            'covariance_type': 'full',
            'random_state': 42,
            'min_regime_duration': 5,  # Minimum days in a regime
            'regime_names': {
                0: 'Bullish',
                1: 'Bearish', 
                2: 'Volatile'
            }
        }
        
        # Alpha Models Configuration
        self.ALPHA_CONFIG = {
            'momentum': {
                'short_window': 10,
                'long_window': 30,
                'breakout_window': 20,
                'volume_threshold': 1.2,
                'rsi_window': 14,
                'signal_threshold': 0.02
            },
            'mean_reversion': {
                'bb_window': 20,
                'bb_std': 2,
                'rsi_window': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'zscore_window': 20,
                'zscore_threshold': 2.0,
                'reversion_threshold': 0.8
            }
        }
        
        # Reinforcement Learning Configuration
        self.RL_CONFIG = {
            'model_type': 'PPO',  # PPO or A2C
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'total_timesteps': 10000,
            'env_config': {
                'transaction_cost': 0.001,
                'max_position': 0.3,
                'lookback_window': 60
            }
        }
        
        # Risk Management Configuration
        self.RISK_CONFIG = {
            'target_volatility': 0.15,  # 15% target volatility
            'max_position': 0.25,       # 25% max position size
            'max_leverage': 1.0,        # No leverage
            'kelly_lookback': 252,      # 1 year for Kelly Criterion
            'var_confidence': 0.05,     # 95% VaR
            'vol_target_window': 60,    # Volatility calculation window
            'position_limit_buffer': 0.05  # 5% buffer for position limits
        }
        
        # Backtesting Configuration
        self.BACKTEST_CONFIG = {
            'initial_capital': 100000,
            'transaction_cost': 0.001,  # 0.1% transaction cost
            'slippage': 0.0001,        # 1 basis point slippage
            'margin_rate': 0.02,       # 2% margin rate
            'min_trade_size': 0.001,   # Minimum trade threshold
            'rebalance_frequency': 'daily'
        }
        
        # Output Configuration
        self.OUTPUT_CONFIG = {
            'base_directory': 'output',
            'save_plots': True,
            'plot_format': 'png',
            'plot_dpi': 300,
            'csv_decimal_places': 6,
            'report_formats': ['csv', 'json', 'html']
        }
        
        # API Configuration
        self.API_CONFIG = {
            'fred_api_key': os.getenv('FRED_API_KEY', ''),
            'yahoo_finance_timeout': 30,
            'max_retries': 3,
            'retry_delay': 1.0
        }
        
        # Logging Configuration
        self.LOGGING_CONFIG = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_logging': True,
            'console_logging': True,
            'log_file': 'output/quant_engine.log'
        }
        
        # Performance Monitoring Configuration
        self.MONITORING_CONFIG = {
            'enable_profiling': False,
            'memory_monitoring': True,
            'performance_alerts': {
                'max_execution_time': 300,  # 5 minutes
                'memory_threshold': 1024,   # 1GB
                'error_threshold': 5
            }
        }
        
        # Dashboard Configuration
        self.DASHBOARD_CONFIG = {
            'port': 5000,
            'host': '0.0.0.0',
            'debug': False,
            'auto_refresh_interval': 30,  # seconds
            'max_chart_points': 1000,
            'default_chart_theme': 'plotly_white'
        }
        
        # Validation Configuration
        self.VALIDATION_CONFIG = {
            'min_data_quality_score': 0.8,
            'max_missing_data_pct': 0.1,
            'outlier_detection_threshold': 3.0,
            'data_staleness_days': 7
        }
    
    def get_regime_model_assignment(self) -> Dict[int, str]:
        """
        Get regime to model assignment mapping
        
        Returns:
            Dictionary mapping regime ID to preferred model
        """
        return {
            0: 'momentum',      # Bullish -> Momentum
            1: 'mean_reversion', # Bearish -> Mean Reversion  
            2: 'momentum'       # Volatile -> Momentum
        }
    
    def get_technical_indicators_config(self) -> Dict[str, Any]:
        """
        Get technical indicators configuration
        
        Returns:
            Technical indicators configuration dictionary
        """
        return {
            'moving_averages': [20, 50, 200],
            'bollinger_bands': {
                'window': 20,
                'std_dev': 2
            },
            'rsi': {
                'window': 14,
                'overbought': 70,
                'oversold': 30
            },
            'macd': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'stochastic': {
                'k_period': 14,
                'd_period': 3
            }
        }
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """
        Get portfolio optimization configuration
        
        Returns:
            Optimization configuration dictionary
        """
        return {
            'objective': 'sharpe_ratio',  # sharpe_ratio, sortino_ratio, calmar_ratio
            'constraints': {
                'max_weight': self.RISK_CONFIG['max_position'],
                'min_weight': -self.RISK_CONFIG['max_position'],
                'leverage_limit': self.RISK_CONFIG['max_leverage'],
                'turnover_limit': 0.5
            },
            'optimization_method': 'efficient_frontier',
            'rebalance_threshold': 0.05,
            'lookback_periods': {
                'returns': 252,
                'volatility': 60,
                'correlation': 126
            }
        }
    
    def validate_config(self) -> Dict[str, List[str]]:
        """
        Validate configuration settings
        
        Returns:
            Dictionary of validation warnings and errors
        """
        warnings = []
        errors = []
        
        # Validate API keys
        if not self.API_CONFIG['fred_api_key']:
            errors.append("FRED_API_KEY environment variable not set")
        
        # Validate data configuration
        if len(self.MARKET_SYMBOLS) == 0:
            errors.append("No market symbols configured")
            
        if len(self.ECONOMIC_INDICATORS) == 0:
            warnings.append("No economic indicators configured")
        
        # Validate risk parameters
        if self.RISK_CONFIG['max_position'] > 1.0:
            warnings.append("Maximum position size exceeds 100%")
            
        if self.RISK_CONFIG['target_volatility'] < 0.05:
            warnings.append("Target volatility might be too low")
            
        # Validate backtest parameters
        if self.BACKTEST_CONFIG['initial_capital'] <= 0:
            errors.append("Initial capital must be positive")
            
        # Validate time parameters
        if self.LOOKBACK_DAYS < 252:
            warnings.append("Lookback period might be insufficient for robust analysis")
        
        return {
            'warnings': warnings,
            'errors': errors,
            'is_valid': len(errors) == 0
        }
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about the current environment
        
        Returns:
            Environment information dictionary
        """
        return {
            'config_version': '1.0.0',
            'python_version': os.sys.version,
            'working_directory': os.getcwd(),
            'environment_variables': {
                'FRED_API_KEY': 'SET' if os.getenv('FRED_API_KEY') else 'NOT_SET'
            },
            'output_directory_exists': os.path.exists(self.OUTPUT_CONFIG['base_directory'])
        }
    
    def update_config(self, section: str, updates: Dict[str, Any]):
        """
        Update configuration section
        
        Args:
            section: Configuration section name
            updates: Dictionary of updates to apply
        """
        if hasattr(self, section.upper() + '_CONFIG'):
            config_dict = getattr(self, section.upper() + '_CONFIG')
            config_dict.update(updates)
        else:
            raise ValueError(f"Configuration section '{section}' not found")
    
    def save_config(self, filepath: str):
        """
        Save current configuration to file
        
        Args:
            filepath: Path to save configuration file
        """
        import json
        
        config_data = {}
        
        # Get all configuration sections
        for attr_name in dir(self):
            if attr_name.endswith('_CONFIG'):
                config_data[attr_name] = getattr(self, attr_name)
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def load_config(self, filepath: str):
        """
        Load configuration from file
        
        Args:
            filepath: Path to configuration file
        """
        import json
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        # Update configuration sections
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name):
                setattr(self, section_name, section_data)
