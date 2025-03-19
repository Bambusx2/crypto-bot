"""
Tests for the trend following strategy.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from cryptobot.exchanges.binance import BinanceExchange
from cryptobot.strategies.trend_following import TrendFollowingStrategy
from cryptobot.risk.manager import RiskManager

@pytest.fixture
def mock_market_data():
    """Create mock market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1H')
    data = {
        'open': np.random.uniform(45000, 55000, len(dates)),
        'high': np.random.uniform(45000, 55000, len(dates)),
        'low': np.random.uniform(45000, 55000, len(dates)),
        'close': np.random.uniform(45000, 55000, len(dates)),
        'volume': np.random.uniform(100, 1000, len(dates))
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def strategy():
    """Create strategy instance for testing."""
    exchange = BinanceExchange('', '', test_mode=True)
    
    config = {
        'position_sizing': {
            'max_position_size': 0.1,
            'risk_per_trade': 0.01
        },
        'portfolio': {
            'max_drawdown': 0.15,
            'max_open_positions': 5,
            'correlation_threshold': 0.7
        }
    }
    
    risk_manager = RiskManager(config)
    
    return TrendFollowingStrategy(
        exchange=exchange,
        risk_manager=risk_manager,
        sma_short=20,
        sma_long=50,
        trend_threshold=0.02
    )

def test_strategy_initialization(strategy):
    """Test strategy initialization."""
    assert strategy.sma_short == 20
    assert strategy.sma_long == 50
    assert strategy.trend_threshold == 0.02
    assert strategy.validate_parameters()

def test_invalid_parameters():
    """Test strategy parameter validation."""
    exchange = BinanceExchange('', '', test_mode=True)
    risk_manager = RiskManager({
        'position_sizing': {'max_position_size': 0.1, 'risk_per_trade': 0.01},
        'portfolio': {'max_drawdown': 0.15, 'max_open_positions': 5, 'correlation_threshold': 0.7}
    })
    
    # Test invalid SMA periods
    with pytest.raises(ValueError):
        TrendFollowingStrategy(
            exchange=exchange,
            risk_manager=risk_manager,
            sma_short=0,
            sma_long=50
        )
    
    with pytest.raises(ValueError):
        TrendFollowingStrategy(
            exchange=exchange,
            risk_manager=risk_manager,
            sma_short=50,
            sma_long=20
        )

def test_signal_generation(strategy, mock_market_data):
    """Test trading signal generation."""
    # Create trending market data
    mock_market_data['close'] = np.linspace(45000, 55000, len(mock_market_data))
    
    signals = strategy.generate_signals(mock_market_data)
    
    assert signals is not None
    assert len(signals) > 0
    
    for signal in signals:
        assert 'type' in signal
        assert 'side' in signal
        assert 'amount' in signal
        assert 'price' in signal
        assert 'stop_loss' in signal
        assert 'take_profit' in signal
        
        assert signal['type'] in ['market', 'limit']
        assert signal['side'] in ['buy', 'sell']
        assert signal['amount'] > 0
        assert signal['price'] > 0
        
        if signal['stop_loss']:
            assert signal['stop_loss'] > 0
        if signal['take_profit']:
            assert signal['take_profit'] > 0

def test_position_sizing(strategy, mock_market_data):
    """Test position size calculation."""
    signal = {
        'price': 50000,
        'side': 'buy'
    }
    
    position_size = strategy.calculate_position_size(signal)
    
    assert position_size > 0
    assert position_size <= 1.0  # Should be a fraction of account balance

def test_metrics_calculation(strategy, mock_market_data):
    """Test strategy metrics calculation."""
    metrics = strategy.calculate_metrics(mock_market_data)
    
    assert 'current_trend' in metrics
    assert metrics['current_trend'] in ['bullish', 'bearish']
    assert 'trend_strength' in metrics
    assert metrics['trend_strength'] >= 0
    assert 'sma_short' in metrics
    assert 'sma_long' in metrics 