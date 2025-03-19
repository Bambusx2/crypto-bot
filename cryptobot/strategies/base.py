"""
Base strategy interface that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import logging

from ..exchanges.base import BaseExchange
from ..risk.manager import RiskManager

class BaseStrategy(ABC):
    """Abstract base class for trading strategy implementations."""
    
    def __init__(self, exchange: BaseExchange, risk_manager: RiskManager, **kwargs):
        """
        Initialize strategy.
        
        Args:
            exchange: Exchange instance for market operations
            risk_manager: Risk manager instance for trade validation
            **kwargs: Strategy-specific parameters
        """
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.parameters = kwargs
    
    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame) -> Optional[List[dict]]:
        """
        Generate trading signals based on market data.
        
        Args:
            market_data: DataFrame containing OHLCV market data
            
        Returns:
            List of signal dictionaries or None if no signals
            Each signal dict should contain:
                - type: str (market, limit)
                - side: str (buy, sell)
                - amount: float
                - price: Optional[float]
                - stop_loss: Optional[float]
                - take_profit: Optional[float]
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: dict) -> float:
        """
        Calculate appropriate position size for a trade.
        
        Args:
            signal: Signal dictionary containing trade details
            
        Returns:
            Position size in base currency
        """
        pass
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        return True
    
    def update_parameters(self, **kwargs):
        """
        Update strategy parameters.
        
        Args:
            **kwargs: New parameter values
        """
        self.parameters.update(kwargs)
        self.validate_parameters()
    
    def get_parameters(self) -> dict:
        """
        Get current strategy parameters.
        
        Returns:
            Dict of parameter names and values
        """
        return self.parameters.copy()
    
    def calculate_metrics(self, market_data: pd.DataFrame) -> dict:
        """
        Calculate strategy-specific metrics.
        
        Args:
            market_data: DataFrame containing OHLCV market data
            
        Returns:
            Dict of metric names and values
        """
        return {}
    
    def ensure_min_notional(self, symbol: str, amount: float, price: float) -> float:
        """
        Ensure the order meets minimum notional value requirements.
        
        Args:
            symbol: Trading pair symbol
            amount: Order amount
            price: Current price
            
        Returns:
            Adjusted amount that meets minimum notional
        """
        min_notional = 5.5  # Reduced from 6.0 to allow more trades while still being safe
        notional = amount * price
        
        if notional < min_notional:
            # Calculate minimum amount needed with 3% buffer (reduced from 5%)
            min_amount = (min_notional / price) * 1.03
            adjusted_amount = max(11, round(min_amount))  # Ensure at least 11 contracts
            logger = logging.getLogger(__name__)
            logger.info(f"Adjusting order amount from {amount} to {adjusted_amount} to meet minimum notional value (current: ${notional:.2f}, required: ${min_notional:.2f})")
            return adjusted_amount
            
        return amount
    
    def update_after_trade(self, symbol: str, is_long: bool) -> None:
        """
        Update strategy state after a trade is executed.
        
        Args:
            symbol: Trading pair symbol
            is_long: Whether the trade was a long position
        """
        # Default implementation does nothing
        pass 