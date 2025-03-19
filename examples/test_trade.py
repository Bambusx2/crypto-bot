"""
Test script to execute a small test trade on Binance Futures with Sonic (S).
"""

import sys
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
import time

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from cryptobot.exchanges.binance import Exchange

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
    
    try:
        # Initialize exchange with futures trading
        logger.info("Initializing exchange connection for futures trading...")
        exchange = Exchange(
            api_key=api_key,
            api_secret=api_secret,
            test_mode=False  # Live mode
        )
        
        # Configure for futures trading
        exchange.client.options['defaultType'] = 'future'
        
        symbol = "S/USDT"
        
        # Check futures balance
        futures_balance = exchange.client.fetch_balance({'type': 'future'})
        usdt_free = float(futures_balance.get('USDT', {}).get('free', 0))
        logger.info(f"Current USDT futures balance: ${usdt_free}")
        
        # Get current S price
        ticker = exchange.client.fetch_ticker(symbol)
        current_price = ticker['last']
        logger.info(f"Current {symbol} price: ${current_price}")
        
        # Calculate position size (minimum $100 notional value required)
        leverage = 10  # 10x leverage
        
        # Set leverage
        try:
            exchange.client.set_leverage(leverage, symbol)
            logger.info(f"Leverage set to {leverage}x")
        except Exception as e:
            logger.warning(f"Could not set leverage: {str(e)}")
        
        # Calculate quantity to meet minimum $100 notional value
        min_notional = 100.0
        contract_qty = round(min_notional / current_price)  # Round to whole number for S
        actual_notional = contract_qty * current_price
        margin_required = actual_notional / leverage
        
        # Ensure we have enough balance
        if usdt_free < margin_required:
            raise Exception(f"Insufficient USDT futures balance. Need ${margin_required:.2f}, have ${usdt_free}")
        
        # Open long position
        logger.info(f"Opening LONG position for {contract_qty} S (${actual_notional:.2f} notional, {leverage}x leverage)...")
        buy_order = exchange.client.create_order(
            symbol=symbol,
            type='MARKET',
            side='BUY',
            amount=contract_qty,
            params={
                'reduceOnly': False,
                'closePosition': False
            }
        )
        logger.info(f"Long position opened: {buy_order}")
        
        # Wait for 10 seconds
        logger.info("Waiting 10 seconds before closing position...")
        time.sleep(10)
        
        # Close position with sell
        logger.info(f"Closing LONG position...")
        sell_order = exchange.client.create_order(
            symbol=symbol,
            type='MARKET',
            side='SELL',
            amount=contract_qty,
            params={
                'reduceOnly': True
            }
        )
        logger.info(f"Position closed: {sell_order}")
        
        # Get final balance
        final_balance = exchange.client.fetch_balance({'type': 'future'})
        final_usdt = float(final_balance.get('USDT', {}).get('free', 0))
        logger.info(f"Final USDT futures balance: ${final_usdt}")
        
        # Calculate PnL
        pnl = final_usdt - usdt_free
        logger.info(f"Trade PnL: ${pnl:+.2f}")
        
        logger.info("Test trade completed!")
        
    except Exception as e:
        logger.error(f"Error during test trade: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 