#!/usr/bin/env python
# execute_direct_trade.py - Directly execute a trade without strategy conditions

import os
import sys
import time
import yaml
import logging
import ccxt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('direct_trade.log')
    ]
)
logger = logging.getLogger('direct_trade')

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None

def execute_direct_trade(symbol="BTC/USDT", side="sell", amount=11, leverage=10):
    """
    Execute a direct trade on the exchange.
    
    Args:
        symbol: Trading pair (default: BTC/USDT)
        side: Trade direction (buy or sell, default: sell)
        amount: Contract size (default: 11)
        leverage: Leverage to use (default: 10)
    """
    logger.info(f"=== EXECUTING DIRECT {side.upper()} TRADE FOR {symbol} ===")
    
    # Load config for API keys
    config_path = 'config/force_trade_now.yml'
    config = load_config(config_path)
    
    if not config:
        logger.error("Cannot load configuration, aborting direct trade")
        return False
    
    # Extract exchange settings
    exchange_settings = config.get('exchange', {})
    api_key = exchange_settings.get('api_key')
    api_secret = exchange_settings.get('api_secret')
    test_mode = exchange_settings.get('test_mode', True)
    
    if not api_key or not api_secret:
        logger.error("API credentials not found in config")
        return False
    
    try:
        # Initialize exchange
        logger.info(f"Initializing exchange (test mode: {test_mode})")
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'test': test_mode
            }
        })
        
        # Set leverage
        logger.info(f"Setting leverage to {leverage}x for {symbol}")
        try:
            exchange.fapiPrivatePostLeverage({
                'symbol': symbol.replace('/', ''),
                'leverage': leverage
            })
        except Exception as e:
            logger.warning(f"Failed to set leverage: {e}, continuing anyway...")
        
        # Get market price
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        logger.info(f"Current price for {symbol}: {current_price}")
        
        # Calculate stop loss and take profit levels
        stop_loss = current_price * (1.01 if side == 'sell' else 0.99)
        take_profit = current_price * (0.97 if side == 'sell' else 1.03)
        
        # Execute market order
        logger.info(f"Placing {side} market order: {amount} contracts of {symbol}")
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=amount
        )
        logger.info(f"Market order placed: {order}")
        
        # Place stop loss order
        logger.info(f"Placing stop loss order at {stop_loss}")
        sl_order = exchange.create_order(
            symbol=symbol,
            type='stop',
            side='buy' if side == 'sell' else 'sell',
            amount=amount,
            price=stop_loss,
            params={'stopPrice': stop_loss}
        )
        logger.info(f"Stop loss order placed: {sl_order}")
        
        # Place take profit order
        logger.info(f"Placing take profit order at {take_profit}")
        tp_order = exchange.create_order(
            symbol=symbol,
            type='take_profit',
            side='buy' if side == 'sell' else 'sell',
            amount=amount,
            price=take_profit,
            params={'stopPrice': take_profit}
        )
        logger.info(f"Take profit order placed: {tp_order}")
        
        logger.info(f"=== DIRECT TRADE EXECUTION COMPLETED SUCCESSFULLY ===")
        return True
        
    except Exception as e:
        logger.error(f"Failed to execute direct trade: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Execute a direct trade')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair (default: BTC/USDT)')
    parser.add_argument('--side', type=str, default='sell', choices=['buy', 'sell'], help='Trade direction (default: sell)')
    parser.add_argument('--amount', type=float, default=11, help='Contract size (default: 11)')
    parser.add_argument('--leverage', type=int, default=10, help='Leverage to use (default: 10)')
    args = parser.parse_args()
    
    # Execute trade with parsed arguments
    result = execute_direct_trade(
        symbol=args.symbol,
        side=args.side,
        amount=args.amount,
        leverage=args.leverage
    )
    
    if result:
        logger.info("Direct trade execution successful")
        sys.exit(0)
    else:
        logger.error("Direct trade execution failed")
        sys.exit(1) 