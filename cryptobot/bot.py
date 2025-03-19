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
import traceback
import json
from pathlib import Path
import pandas as pd

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
        self.order_cooldown = 30  # 30 seconds cooldown between orders for same symbol
        self.position_tracker = {}  # Track positions by symbol
        
        # Add tracking for partial exits
        self.partial_exit_tracker = {}  # Track partial exit status by symbol
        
        # Add performance tracking
        self.trade_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'win_rate': 0.0
        }
        
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        self.bot_stats_file = Path("data/bot_stats.json")
        self._load_stats()
        
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
    
    def _load_stats(self):
        """Load trading statistics."""
        if self.bot_stats_file.exists():
            try:
                with open(self.bot_stats_file, "r") as f:
                    self.trade_stats = json.load(f)
                self.logger.info(f"Loaded trading stats: Win rate {self.trade_stats['win_rate']:.2%}")
            except (json.JSONDecodeError, Exception) as e:
                self.logger.error(f"Error loading stats: {str(e)}")
    
    def _save_stats(self):
        """Save trading statistics."""
        try:
            with open(self.bot_stats_file, "w") as f:
                json.dump(self.trade_stats, f, indent=2)
                f.flush()
        except Exception as e:
            self.logger.error(f"Error saving stats: {str(e)}")
    
    def _update_stats(self, profit: float, is_win: bool):
        """Update trading statistics with new trade result."""
        self.trade_stats['total_trades'] += 1
        
        if is_win:
            self.trade_stats['winning_trades'] += 1
            self.trade_stats['total_profit'] += profit
            self.trade_stats['largest_win'] = max(self.trade_stats['largest_win'], profit)
        else:
            self.trade_stats['losing_trades'] += 1
            self.trade_stats['total_loss'] += abs(profit)
            self.trade_stats['largest_loss'] = max(self.trade_stats['largest_loss'], abs(profit))
        
        # Calculate win rate
        if self.trade_stats['total_trades'] > 0:
            self.trade_stats['win_rate'] = self.trade_stats['winning_trades'] / self.trade_stats['total_trades']
        
        # Save updated stats
        self._save_stats()
        
        # Log performance metrics
        self.logger.info(f"Trading Stats - Win Rate: {self.trade_stats['win_rate']:.2%}, "
                         f"Total Trades: {self.trade_stats['total_trades']}, "
                         f"Net Profit: ${self.trade_stats['total_profit'] - self.trade_stats['total_loss']:.2f}")
    
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
                
                # Dynamic import of strategy class
                try:
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
                except (ImportError, AttributeError) as e:
                    self.logger.error(f"Failed to load strategy {strategy_name}: {str(e)}")
                except Exception as e:
                    self.logger.error(f"Error initializing strategy {strategy_name}: {str(e)}")
    
    def start_trading(self):
        """Start the trading process."""
        if not self.exchange:
            raise RuntimeError("Exchange not initialized")
        
        if not self.strategies:
            raise RuntimeError("No strategies loaded")
        
        self.logger.info("Starting trading process")
        
        # Get loop interval from config or use default
        loop_interval = max(5, self.config.get('trading', {}).get('loop_interval', 10))
        self.logger.info(f"Trading loop interval set to {loop_interval} seconds")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                start_time = time.time()
                self.logger.info(f"Starting trading iteration {iteration}")
                
                # Update portfolio value and metrics
                try:
                    balance = self.exchange.get_balance()
                    # Get total USDT balance from futures account
                    portfolio_value = float(balance.get('USDT', {}).get('total', 0))
                    self.risk_manager.update_portfolio_metrics(portfolio_value)
                    self.logger.info(f"Current portfolio value: ${portfolio_value:.2f}")
                except Exception as e:
                    self.logger.error(f"Error updating portfolio metrics: {str(e)}")
                    self.logger.error(traceback.format_exc())
                
                for symbol in self.config['trading']['quote_currencies']:
                    pair = f"{symbol}/{self.config['trading']['base_currency']}"
                    
                    # Get market data
                    try:
                        # Increase the data limit to ensure we have enough points for all indicators
                        min_data_points = 150  # Ensure enough data for all indicators
                        
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
                        
                        # Track position status
                        old_position_status = self.position_tracker.get(pair, False)
                        self.position_tracker[pair] = has_position
                        
                        # If position was closed since last check, log the event
                        if old_position_status and not has_position:
                            self.logger.info(f"Position for {pair} was closed externally")
                        
                        # Execute strategies
                        all_signals = []
                        for strategy_name, strategy in self.strategies.items():
                            try:
                                self.logger.info(f"Running strategy: {strategy_name} for {pair}")
                                signals = strategy.generate_signals(market_data)
                                
                                if signals:
                                    self.logger.info(f"Strategy {strategy_name} generated signals: {signals}")
                                    # Tag signals with strategy name if not already tagged
                                    for signal in signals:
                                        if 'strategy' not in signal:
                                            signal['strategy'] = strategy_name
                                    all_signals.extend(signals)
                                else:
                                    self.logger.info(f"No signals generated by {strategy_name} for {pair}")
                            except Exception as e:
                                self.logger.error(f"Error in strategy {strategy_name} for {pair}: {str(e)}")
                                self.logger.error(traceback.format_exc())
                        
                        # If multiple signals, prioritize close/reduce signals
                        if all_signals:
                            # First, execute any reduce_only signals
                            close_signals = [s for s in all_signals if s.get('reduce_only', False)]
                            if close_signals:
                                self.logger.info(f"Executing {len(close_signals)} close signals for {pair}")
                                self._execute_signals(pair, close_signals)
                            
                            # Then execute new position signals if no position exists
                            if not has_position:
                                entry_signals = [s for s in all_signals if not s.get('reduce_only', False)]
                                if entry_signals:
                                    # Choose best signal based on strength if multiple
                                    if len(entry_signals) > 1:
                                        # Sort by signal_strength if available
                                        entry_signals.sort(
                                            key=lambda s: s.get('signal_strength', 0) 
                                            if isinstance(s.get('signal_strength'), (int, float)) else 0, 
                                            reverse=True
                                        )
                                        best_signal = entry_signals[0]
                                        self.logger.info(f"Choosing best signal with strength {best_signal.get('signal_strength', 'N/A')} for {pair}")
                                        self._execute_signals(pair, [best_signal])
                                    else:
                                        self.logger.info(f"Executing entry signal for {pair}")
                                        self._execute_signals(pair, entry_signals)
                        
                        # Check for partial take profit opportunities
                        if has_position:
                            self._manage_partial_exits(pair, positions, market_data)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {pair}: {str(e)}")
                        self.logger.error(traceback.format_exc())
                
                # Calculate time spent in this iteration
                iteration_time = time.time() - start_time
                self.logger.info(f"Completed trading iteration {iteration} in {iteration_time:.2f} seconds")
                
                # Sleep for the remaining time in the loop interval
                sleep_time = max(1, loop_interval - iteration_time)
                self.logger.info(f"Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            self.logger.info("Stopping trading process")
            self.cleanup()
    
    def _manage_partial_exits(self, pair: str, positions: List[dict], market_data: pd.DataFrame):
        """
        Manage partial exits for an existing position.
        
        Args:
            pair: Trading pair
            positions: List of position data from exchange
            market_data: Market data DataFrame
        """
        if not self.risk_manager.partial_exits:
            return
        
        try:
            for position in positions:
                contracts = float(position.get('contracts', 0))
                if contracts == 0:
                    continue
                
                entry_price = float(position.get('entryPrice', 0))
                current_price = market_data['close'].iloc[-1]
                position_side = 'long' if contracts > 0 else 'short'
                
                # Calculate profit percentage
                if position_side == 'long':
                    profit_pct = (current_price - entry_price) / entry_price
                else:
                    profit_pct = (entry_price - current_price) / entry_price
                
                # Get partial exit levels from risk manager
                partial_levels = self.risk_manager.calculate_dynamic_stops(
                    pair, entry_price, position_side
                ).get('partial_take_profits', [])
                
                # Track which levels we've already taken
                position_key = f"{pair}_{position.get('id', '')}"
                if position_key not in self.partial_exit_tracker:
                    self.partial_exit_tracker[position_key] = set()
                
                # Check each partial take profit level
                for level in partial_levels:
                    level_price = level['price']
                    size_pct = level['size']
                    
                    # Skip if we've already taken this level
                    level_key = f"{level_price:.8f}"
                    if level_key in self.partial_exit_tracker[position_key]:
                        continue
                    
                    # Check if we've reached this level
                    if (position_side == 'long' and current_price >= level_price) or \
                       (position_side == 'sell' and current_price <= level_price):
                        # Calculate amount to close
                        close_amount = abs(contracts) * size_pct
                        
                        self.logger.info(f"Taking partial profit for {pair} at {profit_pct:.2%} - "
                                       f"closing {size_pct:.0%} ({close_amount} contracts)")
                        
                        try:
                            # Create limit order slightly better than current price
                            close_side = 'sell' if position_side == 'long' else 'buy'
                            limit_price = current_price * (0.999 if position_side == 'long' else 1.001)
                            
                            order = self.exchange.create_order(
                                symbol=pair,
                                order_type='LIMIT',
                                side=close_side,
                                amount=close_amount,
                                price=limit_price,
                                params={
                                    'reduceOnly': True,
                                    'timeInForce': 'GTC',
                                    'workingType': 'MARK_PRICE'
                                }
                            )
                            
                            self.logger.info(f"Placed partial take profit limit order: {order}")
                            
                            # Mark this level as taken
                            self.partial_exit_tracker[position_key].add(level_key)
                            
                            # Track the profit
                            realized_pnl = (limit_price - entry_price) * close_amount if position_side == 'long' \
                                         else (entry_price - limit_price) * close_amount
                            self._update_stats(realized_pnl, realized_pnl > 0)
                            
                        except Exception as e:
                            self.logger.error(f"Error executing partial take profit: {str(e)}")
                            self.logger.error(traceback.format_exc())
        
        except Exception as e:
            self.logger.error(f"Error in partial exit management for {pair}: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def _place_trailing_stop(self, symbol: str, side: str, amount: float, activation_price: float, callback_rate: float) -> Optional[dict]:
        """Place a trailing stop order."""
        try:
            opposite_side = 'buy' if side == 'sell' else 'sell'
            
            params = {
                'type': 'TRAILING_STOP_MARKET',
                'activationprice': activation_price,
                'callbackRate': callback_rate,
                'reduceOnly': True,
                'workingType': 'MARK_PRICE'
            }
            
            order = self.exchange.create_order(
                symbol=symbol,
                order_type='TRAILING_STOP_MARKET',
                side=opposite_side,
                amount=amount,
                params=params
            )
            
            self.logger.info(f"Placed trailing stop: {order}")
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to place trailing stop: {str(e)}")
            return None
    
    def _place_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float) -> Optional[dict]:
        """Place a stop loss order."""
        try:
            opposite_side = 'buy' if side == 'sell' else 'sell'
            
            # Get current price
            ticker = self.exchange.client.fetch_ticker(symbol)
            current_price = float(ticker['last'])
            
            # Validate stop price
            if side == 'buy' and stop_price >= current_price:
                self.logger.warning(f"Stop loss price {stop_price} would trigger immediately for long position. Current price: {current_price}")
                # Adjust stop price to be below current price
                stop_price = current_price * 0.99  # 1% below current price
            elif side == 'sell' and stop_price <= current_price:
                self.logger.warning(f"Stop loss price {stop_price} would trigger immediately for short position. Current price: {current_price}")
                # Adjust stop price to be above current price
                stop_price = current_price * 1.01  # 1% above current price
            
            params = {
                'stopPrice': stop_price,
                'reduceOnly': True,
                'workingType': 'MARK_PRICE'
            }
            
            order = self.exchange.create_order(
                symbol=symbol,
                order_type='STOP_MARKET',
                side=opposite_side,
                amount=amount,
                params=params
            )
            
            self.logger.info(f"Placed stop loss at {stop_price}: {order}")
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to place stop loss: {str(e)}")
            return None
    
    def _place_take_profit(self, symbol: str, side: str, amount: float, take_profit_price: float) -> Optional[dict]:
        """Place a take profit order."""
        try:
            opposite_side = 'buy' if side == 'sell' else 'sell'
            
            # Get current price
            ticker = self.exchange.client.fetch_ticker(symbol)
            current_price = float(ticker['last'])
            
            # Validate take profit price
            if side == 'buy' and take_profit_price <= current_price:
                self.logger.warning(f"Take profit price {take_profit_price} would trigger immediately for long position. Current price: {current_price}")
                # Adjust take profit price to be above current price
                take_profit_price = current_price * 1.02  # 2% above current price
            elif side == 'sell' and take_profit_price >= current_price:
                self.logger.warning(f"Take profit price {take_profit_price} would trigger immediately for short position. Current price: {current_price}")
                # Adjust take profit price to be below current price
                take_profit_price = current_price * 0.98  # 2% below current price
            
            # Place as a limit order
            params = {
                'reduceOnly': True,
                'timeInForce': 'GTC',  # Good Till Cancel
                'workingType': 'MARK_PRICE'
            }
            
            order = self.exchange.create_order(
                symbol=symbol,
                order_type='LIMIT',
                side=opposite_side,
                amount=amount,
                price=take_profit_price,
                params=params
            )
            
            self.logger.info(f"Placed take profit limit order at {take_profit_price}: {order}")
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to place take profit: {str(e)}")
            return None
    
    def _execute_signals(self, pair: str, signals: List[dict]):
        """Execute trading signals for a pair."""
        if not signals:
            return
            
        # Choose best signal based on strength
        signal = signals[0]  # Default to first signal
        signal_strength = signal.get('signal_strength', 0)
        
        for s in signals[1:]:
            if s.get('signal_strength', 0) > signal_strength:
                signal = s
                signal_strength = s.get('signal_strength', 0)
        
        self.logger.info(f"Choosing best signal with strength {signal_strength} for {pair}")
        
        # Validate trade with risk manager
        if not self.risk_manager.validate_trade(pair, signal):
            self.logger.warning(f"Trade validation failed for {pair}")
            return
            
        try:
            # Get current price for validation
            ticker = self.exchange.get_ticker(pair)
            current_price = float(ticker['last'])
            self.logger.info(f"Using current market price for {pair}: {current_price}")
            
            # Ensure minimum notional value
            amount = self.strategies['trend_following'].ensure_min_notional(
                symbol=pair,
                amount=signal['amount'],
                price=current_price
            )
            
            # Prepare order parameters
            params = {}
            
            # Handle stop loss and take profit orders
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            is_long = signal['side'].lower() == 'buy'
            
            # Create main order first
            order = self.exchange.create_order(
                symbol=pair,
                order_type=signal['type'],
                side=signal['side'],
                amount=amount,
                price=signal.get('price'),
                params=params
            )
            
            if order and order.get('id'):
                # Place stop loss order
                if stop_loss:
                    # Ensure stop loss won't trigger immediately
                    if is_long and stop_loss >= current_price:
                        stop_loss = current_price * 0.99  # 1% below for longs
                    elif not is_long and stop_loss <= current_price:
                        stop_loss = current_price * 1.01  # 1% above for shorts
                        
                    stop_order = self._place_stop_loss(
                        symbol=pair,
                        side='sell' if is_long else 'buy',
                        amount=amount,
                        stop_price=stop_loss
                    )
                    if stop_order:
                        self.logger.info(f"Placed stop loss order at {stop_loss}")
                
                # Place take profit order
                if take_profit:
                    # Ensure take profit won't trigger immediately
                    if is_long and take_profit <= current_price:
                        take_profit = current_price * 1.01  # 1% above for longs
                    elif not is_long and take_profit >= current_price:
                        take_profit = current_price * 0.99  # 1% below for shorts
                        
                    tp_order = self._place_take_profit(
                        symbol=pair,
                        side='sell' if is_long else 'buy',
                        amount=amount,
                        take_profit_price=take_profit
                    )
                    if tp_order:
                        self.logger.info(f"Placed take profit order at {take_profit}")
                
                # Update position tracking
                self.open_positions[pair] = {
                    'side': signal['side'],
                    'amount': amount,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'order_id': order['id']
                }
                
        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _calculate_volatility(self, pair: str) -> float:
        """
        Calculate market volatility using standard deviation of returns.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            Volatility as a decimal (e.g., 0.02 for 2% volatility)
        """
        try:
            # Get recent market data (last 24 periods)
            market_data = self.exchange.get_market_data(
                symbol=pair,
                timeframe=self.config['trading']['timeframe'],
                limit=24
            )
            
            if len(market_data) < 2:
                self.logger.warning(f"Insufficient data to calculate volatility for {pair}")
                return 0.02  # Default 2% volatility
            
            # Calculate returns
            returns = market_data['close'].pct_change().dropna()
            
            # Calculate volatility as standard deviation of returns
            volatility = returns.std()
            
            # Annualize volatility based on timeframe
            timeframe = self.config['trading']['timeframe']
            if timeframe == '1m':
                periods_per_day = 1440
            elif timeframe == '5m':
                periods_per_day = 288
            else:
                periods_per_day = 24  # Default to hourly
            
            annualized_volatility = volatility * (periods_per_day ** 0.5)
            
            # Convert to decimal and ensure reasonable bounds
            volatility = min(0.1, max(0.005, annualized_volatility))
            
            self.logger.info(f"Calculated volatility for {pair}: {volatility:.4f}")
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return 0.02  # Default 2% volatility
    
    def cleanup(self):
        """Cleanup resources and perform necessary shutdown operations."""
        # Save final stats
        self._save_stats()
        
        # Close exchange connection
        if self.exchange:
            self.exchange.close()
            
        self.logger.info("Trading bot shutdown complete")

    def _save_trade(self, trade_data: dict):
        """Save trade to JSON file."""
        try:
            # Load existing trades
            trades_file = Path("data/trades.json")
            if trades_file.exists():
                with open(trades_file, "r") as f:
                    try:
                        trades = json.load(f)
                    except json.JSONDecodeError:
                        trades = []
            else:
                trades = []

            # Ensure current time is used for timestamps
            current_time = datetime.utcnow().isoformat()
            
            # Set entry time to current time if not present
            if "entry_time" not in trade_data:
                trade_data["entry_time"] = current_time
            
            # For active trades (no exit), set exit_time to None
            if "exit_time" not in trade_data or not trade_data["exit_time"]:
                trade_data["exit_time"] = None
            
            # Validate and convert numeric fields
            for field in ["entry_price", "exit_price", "amount", "realized_pnl"]:
                try:
                    trade_data[field] = float(trade_data.get(field, 0))
                except (TypeError, ValueError):
                    trade_data[field] = 0.0
            
            # Ensure symbol is properly formatted
            if "symbol" in trade_data:
                trade_data["symbol"] = trade_data["symbol"].strip().upper()
            
            # Add trade to history
            trades.append(trade_data)
            
            # Save updated trades list
            trades_file.parent.mkdir(exist_ok=True)
            with open(trades_file, "w") as f:
                json.dump(trades, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving trade: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc()) 