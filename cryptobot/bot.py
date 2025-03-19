"""
Main bot implementation for cryptocurrency trading.
"""

import logging
import os
from typing import Dict, List, Optional
import yaml
from dotenv import load_dotenv
from datetime import datetime
import time  # Add time import for sleep functionality

from .exchanges.base import BaseExchange
from .strategies.base import BaseStrategy
from .risk.manager import RiskManager

class CryptoBot:
    """Main trading bot class that orchestrates all components."""
    
    def __init__(self, config_path: str):
        """
        Initialize the trading bot.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # Load environment variables
        load_dotenv()
        
        # Set up logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.exchange: Optional[BaseExchange] = None
        self.strategies: Dict[str, BaseStrategy] = {}
        self.risk_manager = None  # Initialize after exchange setup
        
        # Add tracking for last order time to prevent duplicates
        self.last_order_time = {}
        self.order_cooldown = 30  # Increase to 30 seconds cooldown between orders for same symbol
        self.position_tracker = {}  # Track positions by symbol
        
        self.logger.info("CryptoBot initialized with configuration from %s", config_path)
    
    def _setup_logging(self):
        """Configure logging settings."""
        self.logger = logging.getLogger(__name__)
        level = os.getenv('LOG_LEVEL', 'INFO')
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading.log'),
                logging.StreamHandler()
            ]
        )
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict containing configuration
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def initialize_exchange(self):
        """Initialize exchange connection based on configuration."""
        exchange_name = self.config['exchange']['name']
        # Dynamic import of exchange class based on name
        module = __import__(f'cryptobot.exchanges.{exchange_name}', fromlist=['Exchange'])
        exchange_class = getattr(module, 'Exchange')
        
        self.exchange = exchange_class(
            api_key=os.getenv('API_KEY'),
            api_secret=os.getenv('API_SECRET'),
            test_mode=self.config['exchange']['test_mode']
        )
        
        # Configure for futures trading
        self.exchange.client.options['defaultType'] = 'future'
        
        # Set leverage for each symbol
        for symbol in self.config['trading']['quote_currencies']:
            try:
                pair = f"{symbol}/USDT"
                self.exchange.client.set_leverage(
                    self.config['trading']['leverage'],
                    pair
                )
                self.logger.info(f"Set leverage to {self.config['trading']['leverage']}x for {pair}")
            except Exception as e:
                self.logger.warning(f"Could not set leverage for {pair}: {str(e)}")
        
        # Initialize risk manager with exchange
        self.risk_manager = RiskManager(self.config['risk_management'], self.exchange)
        
        self.logger.info(f"Initialized exchange: {exchange_name}")
    
    def load_strategies(self):
        """Load and initialize trading strategies from configuration."""
        for strategy_name, strategy_config in self.config['strategies'].items():
            if strategy_config['enabled']:
                # Log strategy parameters for debugging
                self.logger.info(f"Loading strategy {strategy_name} with config: {strategy_config}")
                self.logger.info(f"Strategy parameters: {strategy_config.get('parameters', {})}")
                
                # Dynamic import of strategy class
                module = __import__(f'cryptobot.strategies.{strategy_name}', fromlist=['Strategy'])
                strategy_class = getattr(module, 'Strategy')
                
                # Create strategy with explicit parameters
                strategy_params = strategy_config.get('parameters', {})
                
                # Clone the parameters to avoid reference issues
                params_copy = dict(strategy_params)
                
                # Convert string 'true'/'false' to boolean values if needed
                for key, value in params_copy.items():
                    if isinstance(value, str) and value.lower() in ['true', 'false']:
                        params_copy[key] = value.lower() == 'true'
                
                self.logger.info(f"Final parameters for {strategy_name}: {params_copy}")
                
                self.strategies[strategy_name] = strategy_class(
                    exchange=self.exchange,
                    risk_manager=self.risk_manager,
                    **params_copy
                )
                
                self.logger.info(f"Loaded strategy: {strategy_name}")
    
    def start_trading(self):
        """Start the trading process."""
        if not self.exchange:
            raise RuntimeError("Exchange not initialized")
        
        if not self.strategies:
            raise RuntimeError("No strategies loaded")
        
        self.logger.info("Starting trading process")
        
        # Get loop interval from config or use default - increase to minimum 10 seconds
        loop_interval = max(10, self.config.get('trading', {}).get('loop_interval', 10))
        self.logger.info(f"Trading loop interval set to {loop_interval} seconds")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                self.logger.info(f"Starting trading iteration {iteration}")
                
                for symbol in self.config['trading']['quote_currencies']:
                    pair = f"{symbol}/{self.config['trading']['base_currency']}"
                    
                    # Get market data
                    try:
                        # Increase the data limit to ensure we have enough points for SMA calculations
                        # We need at least sma_long + a few more points to avoid NaN values
                        sma_long = self.config['strategies']['trend_following']['parameters'].get('sma_long', 15)
                        min_data_points = max(100, sma_long * 3)  # Get at least 3x sma_long or 100 points
                        
                        market_data = self.exchange.get_market_data(
                            pair,
                            timeframe=self.config['trading']['timeframe'],
                            limit=min_data_points
                        )
                        
                        # Add diagnostic information about the market data
                        self.logger.info(f"Fetched {len(market_data)} data points for {pair}")
                        
                        if len(market_data) == 0:
                            self.logger.error(f"No market data received for {pair} - exchange may be experiencing issues")
                            continue
                            
                        # Set the pair name as an attribute of the market_data DataFrame
                        market_data.name = pair
                        
                        # Log the first and last data points for reference
                        if len(market_data) > 0:
                            self.logger.debug(f"First data point: {market_data.iloc[0]}")
                            self.logger.debug(f"Most recent data point: {market_data.iloc[-1]}")
                        
                        # Check for positions before running strategies
                        positions = self.exchange.client.fetch_positions([pair])
                        has_position = any(float(position['contracts']) != 0 for position in positions)
                        if has_position:
                            self.logger.info(f"Current position for {pair}: {positions}")
                            self.position_tracker[pair] = True
                        else:
                            self.position_tracker[pair] = False
                        
                        # Execute strategies
                        for strategy_name, strategy in self.strategies.items():
                            self.logger.info(f"Running strategy: {strategy_name} for {pair}")
                            signals = strategy.generate_signals(market_data)
                            if signals:
                                self.logger.info(f"Generated signals: {signals}")
                                self._execute_signals(pair, signals)
                            else:
                                self.logger.info(f"No signals generated for {pair}")
                                
                    except Exception as e:
                        self.logger.error(f"Error processing {pair}: {str(e)}")
                
                self.logger.info(f"Completed trading iteration {iteration}, sleeping for {loop_interval} seconds")
                # Add delay between iterations to prevent excessive trading
                time.sleep(loop_interval)
        
        except KeyboardInterrupt:
            self.logger.info("Stopping trading process")
            self.cleanup()
    
    def _execute_signals(self, pair: str, signals: List[dict]):
        """
        Execute trading signals while respecting risk management rules.
        
        Args:
            pair: Trading pair
            signals: List of signal dictionaries
        """
        current_time = time.time()
        
        # Check if we're in cooldown period for this symbol
        if pair in self.last_order_time:
            time_since_last_order = current_time - self.last_order_time[pair]
            if time_since_last_order < self.order_cooldown:
                self.logger.info(f"Skipping order for {pair} - in cooldown period ({time_since_last_order:.2f}s < {self.order_cooldown}s)")
                return
        
        # Check if we already have a position for this pair
        try:
            positions = self.exchange.client.fetch_positions([pair])
            has_position = any(float(position['contracts']) != 0 for position in positions)
            
            # For new entry signals, skip if we already have a position
            if has_position and any(not signal.get('reduce_only', False) for signal in signals):
                self.logger.info(f"Skipping new entry signals for {pair} - position already exists")
                # Only keep exit signals
                signals = [signal for signal in signals if signal.get('reduce_only', True)]
                if not signals:
                    return
                
            # Update position tracker
            self.position_tracker[pair] = has_position
            
        except Exception as e:
            self.logger.error(f"Error checking positions: {str(e)}")
        
        for signal in signals:
            if signal.get('reduce_only', False) or self.risk_manager.validate_trade(pair, signal):
                try:
                    # Add reduce_only parameter for position closing
                    params = {
                        'reduceOnly': signal.get('reduce_only', False)
                    }
                    
                    order = self.exchange.create_order(
                        symbol=pair,
                        order_type=signal['type'],
                        side=signal['side'],
                        amount=signal['amount'],
                        price=signal.get('price'),
                        params=params
                    )
                    
                    # Update last order time for this symbol
                    self.last_order_time[pair] = current_time
                    
                    self.logger.info(f"Executed order: {order}")
                    
                    # Build complete trade data
                    trade_data = {
                        'entry_time': datetime.now().isoformat(),
                        'exit_time': datetime.now().isoformat() if signal.get('reduce_only', False) else None,
                        'position_side': signal['side'],
                        'entry_price': float(order['average']),
                        'exit_price': float(order['average']) if signal.get('reduce_only', False) else 0.0,
                        'amount': float(order['amount']),
                        'realized_pnl': float(order.get('info', {}).get('realizedPnl', 0))
                    }
                    
                    # Log the trade
                    for strategy in self.strategies.values():
                        strategy._update_trade_history(trade_data)
                    
                    # Add cooldown after order execution to prevent immediate follow-up orders
                    time.sleep(2)
                    
                    # If this was a new position (not reduce_only), set stop loss and take profit
                    if not signal.get('reduce_only') and signal.get('stop_loss') and signal.get('take_profit'):
                        try:
                            # Place stop loss order
                            stop_loss = self.exchange.create_order(
                                symbol=pair,
                                order_type='stop_market',
                                side='sell' if signal['side'] == 'buy' else 'buy',
                                amount=signal['amount'],
                                price=signal['stop_loss'],
                                params={
                                    'stopPrice': signal['stop_loss'],
                                    'reduceOnly': True
                                }
                            )
                            self.logger.info(f"Placed stop loss: {stop_loss}")
                            
                            # Place take profit order
                            take_profit = self.exchange.create_order(
                                symbol=pair,
                                order_type='take_profit_market',
                                side='sell' if signal['side'] == 'buy' else 'buy',
                                amount=signal['amount'],
                                price=signal['take_profit'],
                                params={
                                    'stopPrice': signal['take_profit'],
                                    'reduceOnly': True
                                }
                            )
                            self.logger.info(f"Placed take profit: {take_profit}")
                            
                        except Exception as e:
                            self.logger.error(f"Failed to place stop loss/take profit: {str(e)}")
                            
                            # Try to close the position if we couldn't set stop loss/take profit
                            try:
                                close_order = self.exchange.create_order(
                                    symbol=pair,
                                    order_type='market',
                                    side='sell' if signal['side'] == 'buy' else 'buy',
                                    amount=signal['amount'],
                                    params={'reduceOnly': True}
                                )
                                self.logger.warning(f"Closed position due to SL/TP failure: {close_order}")
                            except Exception as close_error:
                                self.logger.error(f"Failed to close position: {str(close_error)}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to execute order: {str(e)}")
    
    def cleanup(self):
        """Cleanup resources and perform necessary shutdown operations."""
        if self.exchange:
            self.exchange.close()
        self.logger.info("Trading bot shutdown complete") 