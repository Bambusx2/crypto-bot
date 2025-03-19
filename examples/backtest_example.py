"""
Example script demonstrating how to backtest a trading strategy.
"""

import sys
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import logging

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from cryptobot.exchanges.binance import BinanceExchange
from cryptobot.strategies.trend_following import TrendFollowingStrategy
from cryptobot.risk.manager import RiskManager
from cryptobot.backtesting.engine import BacktestingEngine

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config('config/default_config.yml')
        
        # Initialize components
        exchange = BinanceExchange(
            api_key='',  # Not needed for backtesting
            api_secret='',
            test_mode=True
        )
        
        risk_manager = RiskManager(config['risk_management'])
        
        strategy = TrendFollowingStrategy(
            exchange=exchange,
            risk_manager=risk_manager,
            **config['strategies']['trend_following']['parameters']
        )
        
        # Initialize backtesting engine
        engine = BacktestingEngine(
            exchange=exchange,
            strategy=strategy,
            risk_manager=risk_manager,
            initial_balance=10000.0,
            commission=0.001
        )
        
        # Set up backtest parameters
        symbol = 'BTC/USDT'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        timeframe = '1h'
        
        # Run backtest
        logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
        results = engine.run(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        # Print results
        logger.info("\nBacktest Results:")
        logger.info(f"Initial Balance: ${results['initial_balance']:.2f}")
        logger.info(f"Final Balance: ${results['final_balance']:.2f}")
        logger.info(f"Total Return: {results['total_return']*100:.2f}%")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']*100:.2f}%")
        logger.info(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        # Plot results
        engine.plot_results('data/backtest_results.png')
        logger.info("\nResults plot saved to data/backtest_results.png")
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 