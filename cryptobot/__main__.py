"""
Main entry point for the crypto trading bot.
"""

import argparse
import logging
import sys
from pathlib import Path

from .bot import CryptoBot

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode',
        choices=['live', 'paper'],
        default='paper',
        help='Trading mode (live or paper)'
    )
    
    return parser.parse_args()

def setup_logging():
    """Configure logging settings."""
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main function to run the trading bot."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize the bot
        logger.info(f"Initializing bot with config: {args.config}")
        bot = CryptoBot(args.config)
        
        # Initialize exchange connection
        bot.initialize_exchange()
        
        # Load trading strategies
        bot.load_strategies()
        
        # Start trading
        logger.info(f"Starting bot in {args.mode} mode")
        bot.start_trading()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Bot failed with error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 