"""
Base exchange interface that all exchange implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd

class BaseExchange(ABC):
    """Abstract base class for cryptocurrency exchange implementations."""
    
    def __init__(self, api_key: str, api_secret: str, test_mode: bool = True):
        """
        Initialize exchange connection.
        
        Args:
            api_key: Exchange API key
            api_secret: Exchange API secret
            test_mode: Whether to use test/paper trading mode
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.test_mode = test_mode
    
    @abstractmethod
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch market data for a trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe (e.g., '1m', '1h', '1d')
            limit: Number of candlesticks to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_balance(self, currency: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Get account balance for a specific currency or all currencies.
        
        Args:
            currency: Currency symbol (e.g., 'BTC', 'USDT')
            
        Returns:
            Float balance if currency specified, else dict of currency balances
        """
        pass
    
    @abstractmethod
    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[dict] = None
    ) -> dict:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair symbol
            order_type: Type of order (market, limit)
            side: Order side (buy, sell)
            amount: Order amount
            price: Order price (required for limit orders)
            params: Additional parameters for the order (e.g., reduceOnly, stopPrice)
            
        Returns:
            Dict containing order details
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            symbol: Trading pair symbol
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        """
        Get list of open orders.
        
        Args:
            symbol: Optional trading pair symbol to filter by
            
        Returns:
            List of open order details
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str, symbol: str) -> dict:
        """
        Get status of a specific order.
        
        Args:
            order_id: ID of the order
            symbol: Trading pair symbol
            
        Returns:
            Dict containing order status and details
        """
        pass
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> dict:
        """
        Get current ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict containing current price and 24h statistics
        """
        pass
    
    def close(self):
        """Close exchange connection and cleanup resources."""
        pass 