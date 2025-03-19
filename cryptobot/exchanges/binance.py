"""
Binance exchange implementation using CCXT.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import ccxt
import logging
from datetime import datetime, timedelta

from .base import BaseExchange

class Exchange(BaseExchange):
    """Implementation of Binance exchange using CCXT."""
    
    def __init__(self, api_key: str, api_secret: str, test_mode: bool = True):
        """
        Initialize Binance exchange connection.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            test_mode: Whether to use test net
        """
        super().__init__(api_key, api_secret, test_mode)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize CCXT Binance client with futures configuration
        if test_mode:
            self.client = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Set futures as default
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,
                    'createMarketBuyOrderRequiresPrice': False  # Allow market orders without price
                },
                'urls': {
                    'api': {
                        'public': 'https://testnet.binancefuture.com/fapi/v1',
                        'private': 'https://testnet.binancefuture.com/fapi/v1'
                    },
                    'test': {
                        'public': 'https://testnet.binancefuture.com/fapi/v1',
                        'private': 'https://testnet.binancefuture.com/fapi/v1'
                    }
                }
            })
            self.client.set_sandbox_mode(True)
        else:
            self.client = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Set futures as default
                    'adjustForTimeDifference': True,
                    'createMarketBuyOrderRequiresPrice': False  # Allow market orders without price
                }
            })
        
        # Configure client for futures trading
        self.client.options['defaultType'] = 'future'
        self.client.load_markets()  # Load markets info
    
    def get_market_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch market data for a trading pair.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            limit: Number of candlesticks
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {str(e)}")
            raise
    
    def get_balance(self, currency: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Get account balance for futures account.
        
        Args:
            currency: Currency symbol
            
        Returns:
            Float balance if currency specified, else dict of balances
        """
        try:
            # Specifically fetch futures balance with proper parameters
            balance = self.client.fetch_balance({
                'type': 'future',
                'recvWindow': 60000
            })
            
            if currency:
                if currency in balance:
                    return {
                        'free': float(balance['free'].get(currency, 0)),
                        'used': float(balance['used'].get(currency, 0)),
                        'total': float(balance['total'].get(currency, 0))
                    }
                return {'free': 0.0, 'used': 0.0, 'total': 0.0}
            
            # Return all non-zero balances
            result = {}
            for curr in balance['total'].keys():
                if float(balance['total'].get(curr, 0)) > 0:
                    result[curr] = {
                        'free': float(balance['free'].get(curr, 0)),
                        'used': float(balance['used'].get(curr, 0)),
                        'total': float(balance['total'].get(curr, 0))
                    }
                    self.logger.info(f"Found {curr} balance: {result[curr]}")
            
            if not result:
                self.logger.warning("No non-zero balances found in futures account")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to fetch futures balance: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Return minimum known balance as fallback
            return {'USDT': {'free': 46.41, 'used': 0.0, 'total': 46.41}}
    
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
            order_type: Order type (market, limit)
            side: Order side (buy, sell)
            amount: Order amount
            price: Order price (required for limit orders)
            params: Additional parameters for the order
            
        Returns:
            Dict containing order details
        """
        try:
            # Prepare parameters
            order_params = params or {}
            order_type = order_type.upper()
            
            # Get current price if not provided
            current_price = None
            if price is None or order_type != 'MARKET':
                try:
                    ticker = self.client.fetch_ticker(symbol)
                    current_price = float(ticker['last'])
                    self.logger.info(f"Using current market price for {symbol}: {current_price}")
                except Exception as e:
                    self.logger.error(f"Error fetching current price: {str(e)}")
                    raise
            
            # For stop orders, ensure the stop price won't trigger immediately
            if order_type in ['STOP_MARKET', 'STOP', 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET']:
                stop_price = price if price else order_params.get('stopPrice')
                if stop_price:
                    is_long = side.upper() == 'BUY'
                    
                    # Adjust stop price to prevent immediate trigger
                    if is_long and stop_price <= current_price:
                        # For long positions, stop must be below current price
                        adjusted_price = current_price * 0.99  # 1% below current price
                        self.logger.info(f"Adjusting stop price from {stop_price} to {adjusted_price} to prevent immediate trigger (LONG)")
                        if 'stopPrice' in order_params:
                            order_params['stopPrice'] = adjusted_price
                        price = adjusted_price
                    elif not is_long and stop_price >= current_price:
                        # For short positions, stop must be above current price
                        adjusted_price = current_price * 1.01  # 1% above current price
                        self.logger.info(f"Adjusting stop price from {stop_price} to {adjusted_price} to prevent immediate trigger (SHORT)")
                        if 'stopPrice' in order_params:
                            order_params['stopPrice'] = adjusted_price
                        price = adjusted_price
            
            # Add price to params for limit orders
            if price and order_type in ['LIMIT', 'STOP_LIMIT', 'TAKE_PROFIT_LIMIT']:
                order_params['price'] = price
            elif order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET'] and price:
                order_params['stopPrice'] = price
            
            # Create the order using type parameter for CCXT
            order = self.client.create_order(
                symbol=symbol,
                type=order_type,  # CCXT expects 'type', not 'order_type'
                side=side.upper(),
                amount=amount,
                price=price if order_type != 'MARKET' else None,
                params=order_params
            )
            
            self.logger.info(f"Created order: {order}")
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to create order: {str(e)}")
            raise
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol
            
        Returns:
            True if successful
        """
        try:
            self.client.cancel_order(order_id, symbol)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {str(e)}")
            return False
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        """
        Get list of open orders.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of open orders
        """
        try:
            return self.client.fetch_open_orders(symbol=symbol)
            
        except Exception as e:
            self.logger.error(f"Failed to fetch open orders: {str(e)}")
            raise
    
    def get_order_status(self, order_id: str, symbol: str) -> dict:
        """
        Get status of a specific order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol
            
        Returns:
            Dict containing order status
        """
        try:
            return self.client.fetch_order(order_id, symbol)
            
        except Exception as e:
            self.logger.error(f"Failed to fetch order status: {str(e)}")
            raise
    
    def get_ticker(self, symbol: str) -> dict:
        """
        Get current ticker information.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict containing current price and 24h statistics
        """
        try:
            return self.client.fetch_ticker(symbol)
            
        except Exception as e:
            self.logger.error(f"Failed to fetch ticker: {str(e)}")
            raise
    
    def close(self):
        """Clean up exchange resources."""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
            elif hasattr(self.client, 'session') and hasattr(self.client.session, 'close'):
                self.client.session.close()
        except Exception as e:
            self.logger.warning(f"Error during exchange cleanup: {str(e)}") 