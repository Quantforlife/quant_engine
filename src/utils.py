"""
Utility Functions Module
Common utility functions for the Multi-Modal Regime-Switching Alpha Engine
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None, 
                 console_output: bool = True) -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def ensure_output_dir(output_dir: str = 'output') -> str:
    """
    Ensure output directory exists
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Absolute path to output directory
    """
    abs_path = os.path.abspath(output_dir)
    
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
        logging.getLogger(__name__).info(f"Created output directory: {abs_path}")
    
    return abs_path

def validate_data_quality(data: pd.DataFrame, min_quality_score: float = 0.8) -> Dict[str, Any]:
    """
    Validate data quality and completeness
    
    Args:
        data: DataFrame to validate
        min_quality_score: Minimum acceptable quality score (0-1)
        
    Returns:
        Data quality report dictionary
    """
    if data.empty:
        return {
            'quality_score': 0.0,
            'is_valid': False,
            'issues': ['Data is empty'],
            'missing_data_pct': 100.0,
            'outlier_count': 0,
            'recommendations': ['Verify data source and fetch process']
        }
    
    issues = []
    recommendations = []
    
    # Calculate missing data percentage
    total_cells = data.size
    missing_cells = data.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 100
    
    # Check for excessive missing data
    if missing_pct > 10:
        issues.append(f"High missing data percentage: {missing_pct:.1f}%")
        recommendations.append("Consider data imputation or extending lookback period")
    
    # Check for outliers (using IQR method)
    outlier_count = 0
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if data[col].notna().sum() > 10:  # Need sufficient data points
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            column_outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_count += column_outliers
    
    outlier_pct = (outlier_count / total_cells) * 100 if total_cells > 0 else 0
    
    if outlier_pct > 5:
        issues.append(f"High outlier percentage: {outlier_pct:.1f}%")
        recommendations.append("Consider outlier treatment or data cleaning")
    
    # Check for data staleness
    if hasattr(data.index, 'max') and not data.empty:
        try:
            latest_date = pd.to_datetime(data.index.max())
            days_since_update = (datetime.now() - latest_date).days
            
            if days_since_update > 7:
                issues.append(f"Data is {days_since_update} days old")
                recommendations.append("Update data sources for current information")
        except:
            pass  # Skip if index is not datetime-like
    
    # Check for constant columns
    constant_columns = []
    for col in numeric_columns:
        if data[col].nunique() <= 1:
            constant_columns.append(col)
    
    if constant_columns:
        issues.append(f"Constant columns detected: {constant_columns}")
        recommendations.append("Remove or investigate constant columns")
    
    # Calculate overall quality score
    quality_factors = [
        max(0, (100 - missing_pct) / 100),  # Penalize missing data
        max(0, (100 - outlier_pct) / 100),  # Penalize outliers
        1 if len(constant_columns) == 0 else 0.5,  # Penalize constant columns
        1 if len(data) >= 100 else len(data) / 100  # Reward sufficient data
    ]
    
    quality_score = np.mean(quality_factors)
    is_valid = quality_score >= min_quality_score
    
    return {
        'quality_score': quality_score,
        'is_valid': is_valid,
        'issues': issues,
        'recommendations': recommendations,
        'missing_data_pct': missing_pct,
        'outlier_count': outlier_count,
        'outlier_pct': outlier_pct,
        'constant_columns': constant_columns,
        'data_shape': data.shape,
        'date_range': {
            'start': str(data.index.min()) if not data.empty else None,
            'end': str(data.index.max()) if not data.empty else None
        }
    }

def align_dataframes(dataframes: List[pd.DataFrame], method: str = 'inner') -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to common index
    
    Args:
        dataframes: List of DataFrames to align
        method: Alignment method ('inner', 'outer', 'left', 'right')
        
    Returns:
        List of aligned DataFrames
    """
    if not dataframes:
        return []
    
    # Filter out empty DataFrames
    non_empty_dfs = [df for df in dataframes if not df.empty]
    
    if not non_empty_dfs:
        return dataframes
    
    # Find common index based on method
    if method == 'inner':
        common_index = non_empty_dfs[0].index
        for df in non_empty_dfs[1:]:
            common_index = common_index.intersection(df.index)
    elif method == 'outer':
        common_index = non_empty_dfs[0].index
        for df in non_empty_dfs[1:]:
            common_index = common_index.union(df.index)
    else:
        # Use first DataFrame's index
        common_index = non_empty_dfs[0].index
    
    # Align all DataFrames
    aligned_dfs = []
    for i, df in enumerate(dataframes):
        if df.empty:
            aligned_dfs.append(df)
        else:
            aligned_df = df.reindex(common_index)
            aligned_dfs.append(aligned_df)
    
    return aligned_dfs

def calculate_financial_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate comprehensive financial performance metrics
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        
    Returns:
        Dictionary of financial metrics
    """
    if returns.empty or returns.isna().all():
        return {}
    
    # Clean returns
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    metrics = {}
    
    # Basic return metrics
    metrics['total_return'] = returns.sum()
    metrics['mean_return'] = returns.mean()
    metrics['median_return'] = returns.median()
    metrics['std_return'] = returns.std()
    
    # Annualized metrics (assuming daily returns)
    trading_days_per_year = 252
    
    metrics['annualized_return'] = returns.mean() * trading_days_per_year
    metrics['annualized_volatility'] = returns.std() * np.sqrt(trading_days_per_year)
    
    # CAGR calculation
    if len(returns) > 1:
        cumulative_return = (1 + returns).prod()
        years = len(returns) / trading_days_per_year
        metrics['cagr'] = (cumulative_return ** (1/years)) - 1 if years > 0 else 0
    else:
        metrics['cagr'] = 0
    
    # Risk-adjusted metrics
    excess_returns = returns.mean() * trading_days_per_year - risk_free_rate
    
    if metrics['annualized_volatility'] > 0:
        metrics['sharpe_ratio'] = excess_returns / metrics['annualized_volatility']
    else:
        metrics['sharpe_ratio'] = 0
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1:
        downside_deviation = downside_returns.std() * np.sqrt(trading_days_per_year)
        if downside_deviation > 0:
            metrics['sortino_ratio'] = excess_returns / downside_deviation
        else:
            metrics['sortino_ratio'] = 0
    else:
        metrics['sortino_ratio'] = 0
    
    # Drawdown calculations
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    
    metrics['max_drawdown'] = drawdowns.min()
    metrics['avg_drawdown'] = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0
    
    # Calmar ratio
    if abs(metrics['max_drawdown']) > 0:
        metrics['calmar_ratio'] = metrics['cagr'] / abs(metrics['max_drawdown'])
    else:
        metrics['calmar_ratio'] = 0
    
    # Win rate and other trading metrics
    winning_periods = (returns > 0).sum()
    total_periods = len(returns)
    
    metrics['win_rate'] = winning_periods / total_periods if total_periods > 0 else 0
    metrics['avg_win'] = returns[returns > 0].mean() if (returns > 0).any() else 0
    metrics['avg_loss'] = returns[returns < 0].mean() if (returns < 0).any() else 0
    
    # Profit factor
    total_gains = returns[returns > 0].sum()
    total_losses = abs(returns[returns < 0].sum())
    
    if total_losses > 0:
        metrics['profit_factor'] = total_gains / total_losses
    else:
        metrics['profit_factor'] = float('inf') if total_gains > 0 else 0
    
    # Skewness and Kurtosis
    if len(returns) > 3:
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
    else:
        metrics['skewness'] = 0
        metrics['kurtosis'] = 0
    
    return metrics

def format_performance_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """
    Format performance metrics for display
    
    Args:
        metrics: Dictionary of metrics
        precision: Decimal precision
        
    Returns:
        Dictionary of formatted metrics
    """
    formatted = {}
    
    percentage_metrics = [
        'total_return', 'annualized_return', 'cagr', 'annualized_volatility',
        'max_drawdown', 'avg_drawdown', 'win_rate', 'avg_win', 'avg_loss'
    ]
    
    for key, value in metrics.items():
        if pd.isna(value) or value is None:
            formatted[key] = 'N/A'
        elif key in percentage_metrics:
            formatted[key] = f"{value:.{precision-2}%}"
        elif isinstance(value, float):
            formatted[key] = f"{value:.{precision}f}"
        else:
            formatted[key] = str(value)
    
    return formatted

def save_to_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    Save dictionary to JSON file with proper serialization
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Custom serializer for numpy/pandas objects
    def serialize_object(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return str(obj)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=serialize_object)

def load_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)

def safe_divide(numerator: Union[float, int], denominator: Union[float, int], 
               default: float = 0.0) -> float:
    """
    Perform safe division with default value for zero division
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value when denominator is zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    
    return numerator / denominator

def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in a data series
    
    Args:
        data: Input data series
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    if data.empty:
        return pd.Series(dtype=bool)
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = z_scores > threshold
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return outliers

def create_date_range(start_date: Union[str, datetime], end_date: Union[str, datetime], 
                     freq: str = 'D') -> pd.DatetimeIndex:
    """
    Create pandas DatetimeIndex with specified frequency
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency string ('D', 'B', 'W', 'M', etc.)
        
    Returns:
        DatetimeIndex
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def memory_usage_mb(obj: Any) -> float:
    """
    Calculate memory usage of an object in MB
    
    Args:
        obj: Object to measure
        
    Returns:
        Memory usage in MB
    """
    try:
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum() / (1024 * 1024)
        elif isinstance(obj, pd.Series):
            return obj.memory_usage(deep=True) / (1024 * 1024)
        else:
            import sys
            return sys.getsizeof(obj) / (1024 * 1024)
    except:
        return 0.0

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging
    
    Returns:
        System information dictionary
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'disk_usage_gb': psutil.disk_usage('/').free / (1024**3),
        'current_directory': os.getcwd(),
        'timestamp': datetime.now().isoformat()
    }
