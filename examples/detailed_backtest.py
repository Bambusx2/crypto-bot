"""
Detailed backtesting script with multiple timeframes and parameters.
"""

import sys
from pathlib import Path
import yaml
import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

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

def run_backtest(
    exchange,
    strategy,
    risk_manager,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    initial_balance: float = 10000.0
) -> dict:
    """Run backtest with specified parameters."""
    engine = BacktestingEngine(
        exchange=exchange,
        strategy=strategy,
        risk_manager=risk_manager,
        initial_balance=initial_balance,
        commission=0.001  # 0.1% commission
    )
    
    return engine.run(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )

def plot_results(results_list: list, save_path: str = None):
    """Plot comparison of different backtest results."""
    plt.figure(figsize=(15, 10))
    
    # Plot equity curves
    for result in results_list:
        plt.plot(
            range(len(result['equity_curve'])),
            result['equity_curve'],
            label=f"{result['timeframe']} ({result['total_return']*100:.1f}% return)"
        )
    
    plt.title('Backtest Results Comparison')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

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
        load_dotenv()
        
        # Initialize exchange
        exchange = BinanceExchange(
            api_key=os.getenv('API_KEY'),
            api_secret=os.getenv('API_SECRET'),
            test_mode=True
        )
        
        # Test different timeframes
        timeframes = ['1h', '4h', '1d']
        symbols = ['BTC/USDT', 'ETH/USDT']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 90 days of data
        
        all_results = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"\nTesting {symbol} on {timeframe} timeframe")
                
                # Initialize strategy with timeframe-specific parameters
                if timeframe == '1h':
                    sma_short, sma_long = 20, 50
                elif timeframe == '4h':
                    sma_short, sma_long = 10, 30
                else:  # 1d
                    sma_short, sma_long = 7, 21
                
                strategy = TrendFollowingStrategy(
                    exchange=exchange,
                    risk_manager=RiskManager(config['risk_management']),
                    sma_short=sma_short,
                    sma_long=sma_long,
                    trend_threshold=0.02
                )
                
                # Run backtest
                results = run_backtest(
                    exchange=exchange,
                    strategy=strategy,
                    risk_manager=RiskManager(config['risk_management']),
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Add test parameters to results
                results['timeframe'] = timeframe
                results['symbol'] = symbol
                results['sma_short'] = sma_short
                results['sma_long'] = sma_long
                
                all_results.append(results)
                
                # Print results
                logger.info(f"\nResults for {symbol} - {timeframe}:")
                logger.info(f"Total Return: {results['total_return']*100:.2f}%")
                logger.info(f"Total Trades: {results['total_trades']}")
                logger.info(f"Win Rate: {results['win_rate']*100:.2f}%")
                logger.info(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
                logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        # Create results directory if it doesn't exist
        Path('data').mkdir(exist_ok=True)
        
        # Plot results for each symbol
        for symbol in symbols:
            symbol_results = [r for r in all_results if r['symbol'] == symbol]
            plot_results(
                symbol_results,
                f'data/backtest_results_{symbol.replace("/", "_")}.png'
            )
        
        # Save detailed results to CSV
        results_df = pd.DataFrame([
            {
                'symbol': r['symbol'],
                'timeframe': r['timeframe'],
                'sma_short': r['sma_short'],
                'sma_long': r['sma_long'],
                'total_return': r['total_return'],
                'total_trades': r['total_trades'],
                'win_rate': r['win_rate'],
                'max_drawdown': r['max_drawdown'],
                'sharpe_ratio': r['sharpe_ratio']
            }
            for r in all_results
        ])
        
        results_df.to_csv('data/backtest_summary.csv', index=False)
        logger.info("\nDetailed results saved to data/backtest_summary.csv")
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 