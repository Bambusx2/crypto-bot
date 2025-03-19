"""
Test script to verify exchange connection and basic functionality.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from cryptobot.exchanges.binance import BinanceExchange

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("API key or secret not found in .env file")
        sys.exit(1)
    
    try:
        # Initialize exchange in live mode
        logger.info("Initializing exchange connection...")
        exchange = BinanceExchange(
            api_key=api_key,
            api_secret=api_secret,
            test_mode=False  # Use live mode
        )
        
        # Test market data fetching
        symbol = "BTC/USDT"
        logger.info(f"Fetching market data for {symbol}...")
        market_data = exchange.get_market_data(
            symbol=symbol,
            timeframe='1h',
            limit=10
        )
        logger.info(f"Recent market data:\n{market_data.tail()}\n")
        
        # Test account balance
        logger.info("Fetching account balance...")
        balance = exchange.get_balance()
        logger.info(f"Account balances: {balance}\n")
        
        # Test ticker
        logger.info(f"Fetching current ticker for {symbol}...")
        ticker = exchange.get_ticker(symbol)
        logger.info(f"Current ticker: {ticker}\n")
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 