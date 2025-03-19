"""
Trend following strategy implementation using moving average crossovers.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
import logging
import json
from pathlib import Path
from datetime import datetime
import inspect

from .base import BaseStrategy
from ..exchanges.base import BaseExchange
from ..risk.manager import RiskManager

class Strategy(BaseStrategy):
    """
    Implements a trend following strategy using SMA crossovers.
    Generates buy signals when short SMA crosses above long SMA,
    and sell signals when short SMA crosses below long SMA.
    """
    
    def __init__(
        self,
        exchange: BaseExchange,
        risk_manager: RiskManager,
        sma_short: int = 20,
        sma_long: int = 50,
        trend_threshold: float = 0.02,
        **kwargs
    ):
        """
        Initialize trend following strategy.
        
        Args:
            exchange: Exchange instance for market operations
            risk_manager: Risk manager instance
            sma_short: Short-term SMA period
            sma_long: Long-term SMA period
            trend_threshold: Minimum trend strength threshold
        """
        super().__init__(exchange, risk_manager)
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.trend_threshold = trend_threshold
        self.logger = logging.getLogger(__name__)
        
        # Print all strategy parameters for debugging
        self.logger.info(f"Strategy parameters: {kwargs}")
        self.logger.info(f"allow_default_trades: {kwargs.get('allow_default_trades', False)}")
        self.logger.info(f"force_trades: {kwargs.get('force_trades', False)}")
        self.logger.info(f"debug_mode: {kwargs.get('debug_mode', False)}")
        
        # Trade tracking
        self.current_position = None
        self.last_trade_pnl = 0
        self.total_pnl = 0
        self.trades_history = []
        self.consecutive_losses = 0
        self.waiting_for_exit = False
        
        # Create data directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)
        self.trades_file = Path("data/trades.json")
        self._load_trades()
    
    def _load_trades(self):
        """Load existing trades from file."""
        if self.trades_file.exists():
            with open(self.trades_file, "r") as f:
                self.trades_history = json.load(f)
        else:
            self.trades_history = []

    def _save_trade(self, trade_data: dict):
        """Save trade to JSON file."""
        # Load existing trades first
        try:
            if self.trades_file.exists():
                with open(self.trades_file, "r") as f:
                    self.trades_history = json.load(f)
            else:
                self.trades_history = []
        except json.JSONDecodeError:
            self.trades_history = []

        # Make sure entry_time and exit_time are properly formatted
        if "entry_time" not in trade_data:
            trade_data["entry_time"] = datetime.now().isoformat()
        if "exit_time" not in trade_data:
            trade_data["exit_time"] = datetime.now().isoformat()
            
        # Ensure all required fields are present
        for field in ["position_side", "entry_price", "exit_price", "amount", "realized_pnl"]:
            if field not in trade_data:
                if field == "realized_pnl":
                    trade_data[field] = 0
                elif field in ["entry_price", "exit_price"]:
                    trade_data[field] = 0.0
                elif field == "amount":
                    trade_data[field] = 0.0
                else:
                    trade_data[field] = "unknown"
                    
        # Calculate profit field for display
        if "realized_pnl" in trade_data:
            trade_data["profit"] = float(trade_data["realized_pnl"])
        
        # Add to history
        self.trades_history.append(trade_data)
        
        # Save updated trades list
        with open(self.trades_file, "w") as f:
            json.dump(self.trades_history, f, indent=2)
            f.flush()  # Ensure data is written to disk

    def _check_existing_position(self, symbol: str) -> dict:
        """Check if we have an existing position and its details."""
        try:
            # Ensure symbol is properly formatted with trading pair
            if symbol and '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            # Skip invalid symbols
            if not symbol or symbol.startswith("Unknown") or symbol.startswith("timestamp"):
                self.logger.warning(f"Skipping position check for invalid symbol: {symbol}")
                return None
                
            self.logger.info(f"Checking positions for {symbol}")
            positions = self.exchange.client.fetch_positions([symbol])
            for position in positions:
                if float(position.get('contracts', 0)) != 0:
                    # Extract position information
                    contracts = float(position.get('contracts', 0))
                    pos_info = {
                        'side': 'long' if contracts > 0 else 'short',
                        'amount': abs(contracts),
                        'entry_price': float(position.get('entryPrice', 0)),
                        'unrealized_pnl': float(position.get('unrealizedPnl', 0)),
                        'liquidation_price': float(position.get('liquidationPrice', 0))
                    }
                    self.logger.info(f"Current position for {symbol}: {pos_info}")
                    
                    # Update current_position with the detected position
                    self.current_position = pos_info
                    
                    return pos_info
                    
            # If no position is found, reset current_position
            self.current_position = None
            return None
        except Exception as e:
            self.logger.error(f"Error checking positions for {symbol}: {str(e)}")
            # Don't reset current_position on error to avoid false negative
            return self.current_position

    def _update_trade_history(self, trade_result: dict):
        """Update trade history and adjust strategy based on performance."""
        pnl = trade_result.get('realized_pnl', 0)
        self.last_trade_pnl = pnl
        self.total_pnl += pnl
        
        # Get the symbol for this trade
        symbol = trade_result.get('symbol', 'Unknown')
        
        # Create a complete trade record with all required fields
        trade_data = {
            "entry_time": trade_result.get("entry_time", datetime.now().isoformat()),
            "exit_time": trade_result.get("exit_time", datetime.now().isoformat()),
            "position_side": trade_result.get("position_side", "unknown"),
            "entry_price": float(trade_result.get("entry_price", 0)),
            "exit_price": float(trade_result.get("exit_price", 0)),
            "amount": float(trade_result.get("amount", 0)),
            "realized_pnl": float(pnl),
            "cumulative_pnl": float(self.total_pnl),
            "profit": float(pnl),  # Add profit field for UI display
            "symbol": symbol  # Include symbol in trade record
        }
        
        # Save trade to file - this will reload the trades history first
        self._save_trade(trade_data)
        
        # Keep last 50 trades in memory
        if len(self.trades_history) > 50:
            self.trades_history = self.trades_history[-50:]
        
        # Adjust strategy based on performance
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 3:
                self.trend_threshold *= 1.2  # More conservative
                self.logger.info(f"Adjusting trend threshold up to {self.trend_threshold} after losses for {symbol}")
        else:
            self.consecutive_losses = 0
            if self.trend_threshold > 0.0001:  # Don't get too aggressive
                self.trend_threshold *= 0.9
                self.logger.info(f"Adjusting trend threshold down to {self.trend_threshold} after win for {symbol}")

    def _manage_existing_position(self, symbol: str, current_price: float) -> Optional[List[dict]]:
        """Manage existing position and track its performance."""
        # Skip invalid symbols
        if not symbol or symbol.startswith("Unknown") or symbol.startswith("timestamp"):
            self.logger.warning(f"Skipping position management for invalid symbol: {symbol}")
            return None
            
        position = self._check_existing_position(symbol)
        if not position:
            if self.current_position:
                # Position was closed externally, update history
                self._update_trade_history({
                    'exit_price': current_price,
                    'realized_pnl': self.last_trade_pnl
                })
            self.current_position = None
            self.waiting_for_exit = False
            return None

        # If we're already waiting for an exit, don't generate new exit signals
        if self.waiting_for_exit:
            self.logger.info(f"Already waiting for position exit for {symbol}, skipping signal generation")
            return None

        signals = []
        entry_price = position['entry_price']
        unrealized_pnl = position['unrealized_pnl']
        
        # Calculate profit percentage
        if position['side'] == 'long':
            profit_percentage = (current_price - entry_price) / entry_price
        else:
            profit_percentage = (entry_price - current_price) / entry_price

        # Log position status
        self.logger.info(f"Position for {symbol}: {position['side']}, Entry: {entry_price}, Current: {current_price}, "
                        f"PnL: {unrealized_pnl}, Profit%: {profit_percentage*100:.2f}%")

        # Take profit at 1% or stop loss at 0.5%
        take_profit_threshold = self.parameters.get('take_profit_pct', 0.01)
        stop_loss_threshold = self.parameters.get('stop_loss_pct', 0.005)
        
        if profit_percentage >= take_profit_threshold or profit_percentage <= -stop_loss_threshold:
            message = 'Taking profit' if profit_percentage >= take_profit_threshold else 'Stopping loss'
            self.logger.info(f"{message} at {profit_percentage*100:.2f}% for {symbol}")
            
            # Mark that we're waiting for exit to prevent duplicate close signals
            self.waiting_for_exit = True
            
            signals.append({
                'type': 'MARKET',
                'side': 'sell' if position['side'] == 'long' else 'buy',
                'amount': position['amount'],
                'reduce_only': True
            })
            
            # Record trade result
            self._update_trade_history({
                'entry_price': entry_price,
                'exit_price': current_price,
                'realized_pnl': unrealized_pnl,
                'profit_percentage': profit_percentage,
                'position_side': position['side'],
                'amount': position['amount'],
                'symbol': symbol  # Add symbol to trade history
            })

        return signals

    def calculate_metrics(self, market_data: pd.DataFrame) -> dict:
        """Calculate technical indicators and metrics for trend following strategy."""
        if market_data is None or len(market_data) < max(self.sma_long, self.sma_short) + 5:
            self.logger.warning("Not enough data points for reliable trend analysis")
            return {}
            
        try:
            # Calculate SMAs
            sma_short = market_data['close'].rolling(window=self.sma_short).mean()
            sma_long = market_data['close'].rolling(window=self.sma_long).mean()
            
            # Calculate momentum
            momentum = market_data['close'].pct_change(periods=3)
            
            # Calculate volume trend
            volume_sma = market_data['volume'].rolling(window=10).mean()
            volume_trend = market_data['volume'] / volume_sma
            
            # Calculate price trend strength
            current_price = market_data['close'].iloc[-1]
            price_change = market_data['close'].pct_change()
            trend_strength = price_change.rolling(window=5).mean()
            
            # Calculate signal strengths
            long_signal_strength = 0.0
            short_signal_strength = 0.0
            
            # Price above/below SMAs (40% weight)
            if current_price > sma_short.iloc[-1] > sma_long.iloc[-1]:
                long_signal_strength += 0.4 * min(1.0, (current_price - sma_long.iloc[-1]) / current_price * 100)
            elif current_price < sma_short.iloc[-1] < sma_long.iloc[-1]:
                short_signal_strength += 0.4 * min(1.0, (sma_long.iloc[-1] - current_price) / current_price * 100)
            
            # Momentum component (30% weight)
            if momentum.iloc[-1] > 0:
                long_signal_strength += 0.3 * min(1.0, momentum.iloc[-1] * 100)
            else:
                short_signal_strength += 0.3 * min(1.0, abs(momentum.iloc[-1] * 100))
            
            # Volume trend component (30% weight)
            if volume_trend.iloc[-1] > 1.0:
                if long_signal_strength > 0:
                    long_signal_strength += 0.3 * min(1.0, volume_trend.iloc[-1] - 1.0)
                if short_signal_strength > 0:
                    short_signal_strength += 0.3 * min(1.0, volume_trend.iloc[-1] - 1.0)
            
            return {
                'price': current_price,
                'sma_short': sma_short.iloc[-1],
                'sma_long': sma_long.iloc[-1],
                'momentum': momentum.iloc[-1],
                'volume_trend': volume_trend.iloc[-1],
                'trend_strength': trend_strength.iloc[-1],
                'long_signal_strength': long_signal_strength,
                'short_signal_strength': short_signal_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trend metrics: {str(e)}")
            return {}

    def generate_signals(self, market_data: pd.DataFrame) -> Optional[List[dict]]:
        """Generate trading signals based on market data."""
        # Check if we have any data at all
        if market_data is None or len(market_data) == 0:
            self.logger.error("No market data available - cannot generate signals")
            return None
            
        current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else None
        if current_price is None:
            self.logger.error("No price data available")
            return None
        
        # Extract the symbol from market data
        if hasattr(market_data, 'name') and market_data.name and '/' in market_data.name:
            symbol = market_data.name
        else:
            # Try to extract symbol using various methods (code unchanged)
            # ...
            # Final fallback
            symbol = "Unknown/USDT"
            
        # Ensure the symbol has no spaces or unexpected characters
        symbol = symbol.strip()
        
        # Set name in market_data for future reference
        market_data.name = symbol
            
        self.logger.info(f"Analyzing {symbol} at price {current_price}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(market_data)
        if not metrics:
            self.logger.warning(f"Could not calculate metrics for {symbol}")
            return None
        
        # Get signal strengths and other metrics
        long_signal_strength = metrics.get('long_signal_strength', 0)
        short_signal_strength = metrics.get('short_signal_strength', 0)
        trend_strength = metrics.get('trend_strength', 0)
        volume_ratio = metrics.get('volume_trend', 0)
        
        # Calculate volatility for dynamic stop loss and take profit
        volatility = market_data['close'].pct_change().std()
        atr = market_data['high'].rolling(14).max() - market_data['low'].rolling(14).min()
        avg_atr = atr.mean() / current_price  # As percentage of price
        
        # Use max of volatility or ATR, with minimum of 0.5%
        price_movement = max(volatility, avg_atr, 0.005)
        
        # Generate signals with proper stop loss and take profit levels
        signals = []
        
        # LONG SIGNAL
        if long_signal_strength > self.trend_threshold:
            # Strong bullish signal
            self.logger.info(f"STRONG LONG signal for {symbol} (strength: {long_signal_strength:.4f})")
            
            # Calculate position size
            position_size = self.calculate_position_size({
                'price': current_price,
                'side': 'buy',
                'symbol': symbol,
                'signal_strength': long_signal_strength
            })
            
            if position_size > 0:
                # Calculate stop loss and take profit based on volatility
                stop_loss = current_price * (1 - price_movement * 2)  # 2x volatility for stop loss
                take_profit = current_price * (1 + price_movement * 3)  # 3x volatility for take profit
                
                self.logger.info(f"Generating LONG signal for {symbol}: Price={current_price}, Size={position_size}, "
                                f"SL={stop_loss:.8f} ({((stop_loss/current_price)-1)*100:.2f}%), "
                                f"TP={take_profit:.8f} ({((take_profit/current_price)-1)*100:.2f}%)")
                
                signals.append({
                    'type': 'MARKET',
                    'side': 'buy',
                    'amount': position_size,
                    'price': None,  # Market order
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
        
        # SHORT SIGNAL
        elif short_signal_strength > self.trend_threshold:
            # Strong bearish signal
            self.logger.info(f"STRONG SHORT signal for {symbol} (strength: {short_signal_strength:.4f})")
            
            # Calculate position size
            position_size = self.calculate_position_size({
                'price': current_price,
                'side': 'sell',
                'symbol': symbol,
                'signal_strength': short_signal_strength
            })
            
            if position_size > 0:
                # Calculate stop loss and take profit based on volatility
                stop_loss = current_price * (1 + price_movement * 2)  # 2x volatility for stop loss
                take_profit = current_price * (1 - price_movement * 3)  # 3x volatility for take profit
                
                self.logger.info(f"Generating SHORT signal for {symbol}: Price={current_price}, Size={position_size}, "
                                f"SL={stop_loss:.8f} ({((stop_loss/current_price)-1)*100:.2f}%), "
                                f"TP={take_profit:.8f} ({((take_profit/current_price)-1)*100:.2f}%)")
                
                signals.append({
                    'type': 'MARKET',
                    'side': 'sell',
                    'amount': position_size,
                    'price': None,  # Market order
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
        
        else:
            self.logger.info(f"No signals generated for {symbol} - insufficient signal strength")
        
        return signals if signals else None

    def calculate_position_size(self, signal: dict) -> float:
        """Calculate position size with improved risk management."""
        price = signal.get('price', 0)
        side = signal.get('side', 'buy')
        symbol = signal.get('symbol', 'Unknown/USDT')
        signal_strength = signal.get('signal_strength', 0.5)  # Default to medium strength
        
        if price <= 0:
            self.logger.error(f"Invalid price for position sizing: {price}")
            return 0
            
        # Get risk parameters
        max_position_size = self.risk_manager.config.get('position_sizing', {}).get('max_position_size', 50)
        risk_per_trade = self.risk_manager.config.get('position_sizing', {}).get('risk_per_trade', 0.02)
        
        # Get account balance
        try:
            balance = self.exchange.get_balance()
            usdt_balance = float(balance.get('USDT', {}).get('free', 0))
            self.logger.info(f"Account balance: {usdt_balance} USDT")
        except Exception as e:
            self.logger.error(f"Error getting balance: {str(e)}")
            usdt_balance = 1000  # Default fallback
        
        # Scale position size with account balance and risk percentage
        risk_amount = usdt_balance * risk_per_trade
        
        # Scale with signal strength (stronger signals = larger positions)
        if signal_strength > 0:
            risk_amount = risk_amount * (0.5 + signal_strength)
        
        # Dynamic risk based on volatility
        # For higher priced assets, lower the contract size to manage risk
        if 'BTC' in symbol:
            price_factor = 0.2  # BTC is expensive, use smaller positions
        elif any(coin in symbol for coin in ['ETH', 'AVAX']):
            price_factor = 0.4  # Medium price coins
        else:
            price_factor = 0.8  # Lower price coins
        
        # Base contract calculation (notional value)
        contract_value = price * 1  # Value of a single contract
        
        # Calculate position size based on risk amount and contract value
        position_size = (risk_amount / contract_value) * price_factor
        
        # Scale by leverage (if using futures)
        leverage = 10  # Default leverage
        position_size = position_size * leverage
        
        # Round to appropriate precision
        if price < 1:
            # For cheaper coins, round to 0 decimals (whole coins)
            position_size = round(position_size)
        else:
            # For more expensive coins, round to 1 decimal
            position_size = round(position_size, 1)
        
        # Enforce minimum and maximum positions
        position_size = max(11, min(position_size, max_position_size))
        
        self.logger.info(f"Calculated position size for {symbol}: {position_size} contracts (signal strength: {signal_strength:.2f})")
        
        return position_size
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        if not isinstance(self.sma_short, int) or self.sma_short <= 0:
            return False
        if not isinstance(self.sma_long, int) or self.sma_long <= 0:
            return False
        if self.sma_short >= self.sma_long:
            return False
        if not isinstance(self.trend_threshold, (int, float)) or self.trend_threshold <= 0:
            return False
        return True 