"""
Data Fetcher Module
Handles fetching market data from Yahoo Finance and economic data from FRED API
"""

import pandas as pd
import numpy as np
import yfinance as yf
import fredapi
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

class DataFetcher:
    """
    Fetches market and economic data from various sources
    """
    
    def __init__(self, fred_api_key: str):
        """
        Initialize data fetcher with API credentials
        
        Args:
            fred_api_key: FRED API key for economic data
        """
        self.logger = logging.getLogger(__name__)
        self.fred = fredapi.Fred(api_key=fred_api_key)
        
    def fetch_market_data(self, symbols: List[str], start_date: datetime, 
                         end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data from Yahoo Finance
        
        Args:
            symbols: List of symbols to fetch (e.g., ['^GSPC', '^IXIC'])
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary with symbol as key and price DataFrame as value
        """
        self.logger.info(f"Fetching market data for symbols: {symbols}")
        
        market_data = {}
        
        for symbol in symbols:
            try:
                self.logger.info(f"Downloading {symbol}...")
                ticker = yf.Ticker(symbol)
                
                # Fetch historical data
                hist_data = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d'
                )
                
                if hist_data.empty:
                    self.logger.warning(f"No data found for {symbol}")
                    continue
                
                # Calculate additional technical indicators
                hist_data['Returns'] = hist_data['Close'].pct_change()
                hist_data['LogReturns'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
                hist_data['Volatility'] = hist_data['Returns'].rolling(window=20).std()
                
                # Moving averages
                hist_data['MA_20'] = hist_data['Close'].rolling(window=20).mean()
                hist_data['MA_50'] = hist_data['Close'].rolling(window=50).mean()
                hist_data['MA_200'] = hist_data['Close'].rolling(window=200).mean()
                
                # Bollinger Bands
                bb_window = 20
                bb_std = 2
                hist_data['BB_Middle'] = hist_data['Close'].rolling(window=bb_window).mean()
                hist_data['BB_Upper'] = hist_data['BB_Middle'] + (
                    hist_data['Close'].rolling(window=bb_window).std() * bb_std
                )
                hist_data['BB_Lower'] = hist_data['BB_Middle'] - (
                    hist_data['Close'].rolling(window=bb_window).std() * bb_std
                )
                
                # RSI
                hist_data['RSI'] = self._calculate_rsi(hist_data['Close'])
                
                market_data[symbol] = hist_data.dropna()
                self.logger.info(f"Successfully fetched {len(hist_data)} records for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return market_data
    
    def fetch_economic_data(self, indicators: List[str], start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        """
        Fetch economic indicators from FRED
        
        Args:
            indicators: List of FRED series IDs
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with economic indicators
        """
        self.logger.info(f"Fetching economic data for indicators: {indicators}")
        
        economic_data = pd.DataFrame()
        
        for indicator in indicators:
            try:
                self.logger.info(f"Downloading {indicator}...")
                
                # Fetch data from FRED
                series_data = self.fred.get_series(
                    indicator,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if series_data.empty:
                    self.logger.warning(f"No data found for {indicator}")
                    continue
                
                # Add to economic data DataFrame
                economic_data[indicator] = series_data
                self.logger.info(f"Successfully fetched {len(series_data)} records for {indicator}")
                
            except Exception as e:
                self.logger.error(f"Error fetching {indicator}: {str(e)}")
                continue
        
        # Forward fill missing values and interpolate
        economic_data = economic_data.fillna(method='ffill').interpolate()
        
        # Calculate changes and trends
        for col in economic_data.columns:
            economic_data[f'{col}_change'] = economic_data[col].pct_change()
            economic_data[f'{col}_ma'] = economic_data[col].rolling(window=12).mean()
        
        self.logger.info(f"Economic data shape: {economic_data.shape}")
        return economic_data.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Price series
            window: RSI calculation window
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_data_summary(self, market_data: Dict[str, pd.DataFrame], 
                        economic_data: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for fetched data
        
        Args:
            market_data: Market data dictionary
            economic_data: Economic data DataFrame
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'market_data_summary': {},
            'economic_data_summary': {},
            'data_quality': {}
        }
        
        # Market data summary
        for symbol, data in market_data.items():
            summary['market_data_summary'][symbol] = {
                'records': len(data),
                'start_date': data.index.min().strftime('%Y-%m-%d'),
                'end_date': data.index.max().strftime('%Y-%m-%d'),
                'mean_volume': data['Volume'].mean(),
                'mean_return': data['Returns'].mean(),
                'volatility': data['Returns'].std()
            }
        
        # Economic data summary
        if not economic_data.empty:
            summary['economic_data_summary'] = {
                'indicators': list(economic_data.columns),
                'records': len(economic_data),
                'start_date': economic_data.index.min().strftime('%Y-%m-%d'),
                'end_date': economic_data.index.max().strftime('%Y-%m-%d')
            }
        
        return summary
