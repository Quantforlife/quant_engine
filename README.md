# Multi-Modal Regime-Switching Alpha Engine

## Overview

This is a comprehensive quantitative finance platform that implements a multi-modal regime-switching alpha engine for algorithmic trading. The system combines Hidden Markov Models for market regime detection, multiple alpha generation strategies (momentum and mean reversion), reinforcement learning for portfolio optimization, and comprehensive risk management. The platform provides both a command-line interface for backtesting and a Streamlit web dashboard for interactive monitoring and visualization.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Components Architecture

**Data Layer**
- Uses Yahoo Finance API for market data (S&P 500, NASDAQ indices)
- Integrates FRED API for economic indicators (CPI, GDP, unemployment rate, 10-year Treasury)
- Implements robust data fetching with error handling and validation
- Stores data in pandas DataFrames for efficient processing

**Regime Detection Engine**
- Implements Hidden Markov Models using hmmlearn library
- Detects 3 market regimes: Bullish, Bearish, and Volatile
- Uses standardized features combining market returns and economic indicators
- Provides regime probabilities and state transitions

**Alpha Generation Models**
- Momentum Strategy: Uses moving averages, breakouts, and volume analysis
- Mean Reversion Strategy: Implements Bollinger Bands, RSI, and z-score analysis
- Regime-aware model selection (momentum for bullish/volatile, mean reversion for bearish)
- Generates standardized alpha signals across different time horizons

**Reinforcement Learning Portfolio Manager**
- Custom OpenAI Gym environment for portfolio optimization
- Supports PPO and A2C algorithms from Stable Baselines3
- Optimizes portfolio weights based on alpha signals and market conditions
- Handles transaction costs and position constraints

**Risk Management System**
- Implements Kelly Criterion for position sizing
- Volatility targeting with configurable risk parameters
- Position limits and leverage controls
- Real-time risk metric calculation and monitoring

**Backtesting Engine**
- Comprehensive backtesting with transaction costs and slippage
- Performance analytics including Sharpe ratio, maximum drawdown, and volatility
- Trade-level analysis and regime-based performance attribution
- Monte Carlo simulations for robust statistical analysis

**Reporting and Visualization**
- Automated CSV export with test numbering system
- Streamlit dashboard for interactive visualization
- Performance charts, regime visualization, and risk metrics
- Trade reporting with detailed analytics

### Design Patterns

**Modular Architecture**: Each component is encapsulated in separate classes with clear interfaces, enabling easy testing and maintenance.

**Configuration-Driven**: Central configuration management through Config class, allowing easy parameter adjustments without code changes.

**Event-Driven Processing**: Pipeline architecture where each component processes data and passes results to the next stage.

**Error Handling**: Comprehensive logging and graceful error handling throughout the system.

## External Dependencies

### APIs and Data Sources
- **Yahoo Finance API**: Primary source for market data (stock indices, prices, volumes)
- **FRED API**: Federal Reserve Economic Data for macroeconomic indicators
- **Environment Variable**: FRED_API_KEY required for economic data access

### Python Libraries
- **Data Processing**: pandas, numpy for data manipulation and numerical computing
- **Machine Learning**: scikit-learn for preprocessing, hmmlearn for Hidden Markov Models
- **Reinforcement Learning**: stable-baselines3, gymnasium for RL portfolio optimization
- **Visualization**: plotly for interactive charts, streamlit for web dashboard
- **Finance**: yfinance for market data, fredapi for economic data

### Infrastructure
- **File System**: Local file storage for outputs, logs, and configuration
- **Logging**: Python logging module for comprehensive system monitoring
- **Output Management**: Automated CSV generation with incremental test numbering

### Optional Dependencies
- Reinforcement learning components are optional (graceful fallback if not available)
- Dashboard can run independently of the main backtesting pipeline
- Economic data integration is configurable based on API availability
