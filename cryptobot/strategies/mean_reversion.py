"""
Mean reversion strategy implementation using RSI and price reversion to the mean.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime

from .base import BaseStrategy
from ..exchanges.base import BaseExchange
from ..risk.manager import RiskManager

class Strategy(BaseStrategy):
    """
    Implements a mean reversion strategy using RSI and price deviation from mean.
    Generates buy signals when price is oversold and sell signals when price is overbought.
    """
    
    def __init__(
        self,
        exchange: BaseExchange,
        risk_manager: RiskManager,
        rsi_period: int = 5,
        overbought: int = 70,
        oversold: int = 30,
        mean_period: int = 8,
        mean_threshold: float = 0.005,
        use_with_trend: bool = True,
        **kwargs
    ):
        """
        Initialize mean reversion strategy.
        
        Args:
            exchange: Exchange instance for market operations
            risk_manager: Risk manager instance
            rsi_period: Period for RSI calculation
            overbought: RSI level considered overbought
            oversold: RSI level considered oversold
            mean_period: Period for mean calculation
            mean_threshold: Minimum deviation from mean to generate signals
            use_with_trend: Only take mean reversion signals in the direction of the overall trend
        """
        super().__init__(exchange, risk_manager)
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.mean_period = mean_period
        self.mean_threshold = mean_threshold
        self.use_with_trend = use_with_trend
        self.parameters = kwargs
        self.logger = logging.getLogger(__name__)
        
        # Print all strategy parameters for debugging
        self.logger.info(f"Mean Reversion Strategy parameters: {self.parameters}")
        
        # Trade tracking
        self.current_position = None
        self.last_trade_pnl = 0
        self.total_pnl = 0
        self.trades_history = []
        self.consecutive_losses = 0
        self.waiting_for_exit = False
        
        # Create data directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)
        self.trades_file = Path("data/mean_reversion_trades.json")
        self._load_trades()
    
    def _load_trades(self):
        """Load existing trades from file."""
        if self.trades_file.exists():
            try:
                with open(self.trades_file, "r") as f:
                    self.trades_history = json.load(f)
            except json.JSONDecodeError:
                self.trades_history = []
        else:
            self.trades_history = []

    def _save_trade(self, trade_data: dict):
        """Save trade to JSON file."""
        # Load existing trades first
        try:
            if self.trades_file.exists():
                try:
                    with open(self.trades_file, "r") as f:
                        self.trades_history = json.load(f)
                except json.JSONDecodeError:
                    self.trades_history = []
        except Exception:
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
        self.logger.info(f"Mean Reversion Position for {symbol}: {position['side']}, Entry: {entry_price}, Current: {current_price}, "
                        f"PnL: {unrealized_pnl}, Profit%: {profit_percentage*100:.2f}%")

        # Take profit at configured percentage or stop loss at configured percentage
        take_profit_threshold = self.parameters.get('take_profit_pct', 0.02)
        stop_loss_threshold = self.parameters.get('stop_loss_pct', 0.008)
        
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
        else:
            self.consecutive_losses = 0

    def calculate_metrics(self, market_data: pd.DataFrame) -> dict:
        """Calculate technical indicators and metrics for mean reversion strategy."""
        if market_data is None or len(market_data) < max(self.rsi_period, self.mean_period) + 5:
            self.logger.warning("Not enough data points for reliable mean reversion analysis")
            return {}
            
        try:
            # Calculate RSI
            delta = market_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate mean and deviation
            mean_price = market_data['close'].rolling(window=self.mean_period).mean()
            price_deviation = (market_data['close'] - mean_price) / mean_price
            
            # Calculate additional indicators
            std_dev = market_data['close'].rolling(window=self.mean_period).std()
            z_score = (market_data['close'] - mean_price) / std_dev
            
            # Bollinger Bands
            bb_middle = mean_price
            bb_upper = bb_middle + (std_dev * 2)
            bb_lower = bb_middle - (std_dev * 2)
            
            # Volume analysis
            volume_sma = market_data['volume'].rolling(window=10).mean()
            volume_ratio = market_data['volume'] / volume_sma
            
            # Trend analysis - determine if we're in a significant trend
            sma_short = market_data['close'].rolling(window=5).mean()
            sma_long = market_data['close'].rolling(window=15).mean()
            trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
            trend_direction = 1 if trend_strength > 0 else -1  # 1 for up, -1 for down
            
            # Current values
            current_price = market_data['close'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_mean = mean_price.iloc[-1]
            current_deviation = price_deviation.iloc[-1]
            current_z_score = z_score.iloc[-1]
            current_volume_ratio = volume_ratio.iloc[-1]
            
            # Mean reversion signal strength (0.0 to 1.0)
            long_signal_strength = 0.0
            short_signal_strength = 0.0
            
            # RSI component (40%)
            if current_rsi < self.oversold:
                # Oversold - bullish signal
                long_signal_strength += 0.4 * (1.0 - (current_rsi / self.oversold))
            elif current_rsi > self.overbought:
                # Overbought - bearish signal
                short_signal_strength += 0.4 * ((current_rsi - self.overbought) / (100 - self.overbought))
            
            # Deviation component (40%)
            if current_deviation < -self.mean_threshold:
                # Price below mean - bullish signal
                long_signal_strength += 0.4 * min(1.0, abs(current_deviation) / (self.mean_threshold * 3))
            elif current_deviation > self.mean_threshold:
                # Price above mean - bearish signal
                short_signal_strength += 0.4 * min(1.0, current_deviation / (self.mean_threshold * 3))
            
            # Volume component (20%)
            if current_volume_ratio > 1.2:
                # High volume confirms mean reversion signals
                if long_signal_strength > 0:
                    long_signal_strength += 0.2 * min(1.0, current_volume_ratio - 1.0)
                if short_signal_strength > 0:
                    short_signal_strength += 0.2 * min(1.0, current_volume_ratio - 1.0)
            
            # If use_with_trend is enabled, adjust signal strengths based on trend direction
            if self.use_with_trend and abs(trend_strength) > 0.002:
                if trend_direction < 0:  # Downtrend
                    long_signal_strength *= 0.5  # Reduce long signals in downtrend
                elif trend_direction > 0:  # Uptrend
                    short_signal_strength *= 0.5  # Reduce short signals in uptrend
            
            return {
                'price': current_price,
                'rsi': current_rsi,
                'mean_price': current_mean,
                'price_deviation': current_deviation,
                'z_score': current_z_score,
                'bb_upper': bb_upper.iloc[-1],
                'bb_lower': bb_lower.iloc[-1],
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'volume_ratio': current_volume_ratio,
                'long_signal_strength': long_signal_strength,
                'short_signal_strength': short_signal_strength
            }
        except Exception as e:
            self.logger.error(f"Error calculating mean reversion metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}

    def generate_signals(self, market_data: pd.DataFrame) -> Optional[List[dict]]:
        """Generate trading signals based on mean reversion strategy."""
        # Check if we have any data at all
        if market_data is None or len(market_data) == 0:
            self.logger.error("No market data available - cannot generate mean reversion signals")
            return None
            
        current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else None
        if current_price is None:
            self.logger.error("No price data available for mean reversion strategy")
            return None
        
        # Extract the symbol from market data
        if hasattr(market_data, 'name') and market_data.name and '/' in market_data.name:
            symbol = market_data.name
        else:
            # Try to extract symbol using various methods
            if 'symbol' in market_data.columns:
                symbol = market_data['symbol'].iloc[0]
            else:
                # Final fallback
                symbol = "Unknown/USDT"
            
        # Ensure the symbol has no spaces or unexpected characters
        symbol = symbol.strip()
        
        # Set name in market_data for future reference
        market_data.name = symbol
            
        self.logger.info(f"[Mean Reversion] Analyzing {symbol} at price {current_price}")
        
        # First, check and manage existing positions
        existing_position = self._check_existing_position(symbol)
        
        # Return position management signals if any
        position_signals = self._manage_existing_position(symbol, current_price)
        if position_signals:
            return position_signals
        
        # Don't generate new signals if we have an existing position or are waiting for exit
        if existing_position or self.waiting_for_exit:
            self.logger.info(f"Not generating new mean reversion signals: existing position={bool(existing_position)}, waiting_for_exit={self.waiting_for_exit}")
            return None
        
        # Calculate metrics
        metrics = self.calculate_metrics(market_data)
        if not metrics:
            self.logger.warning(f"Could not calculate mean reversion metrics for {symbol}")
            return None
        
        # Get signal strengths and other metrics
        long_signal_strength = metrics.get('long_signal_strength', 0)
        short_signal_strength = metrics.get('short_signal_strength', 0)
        current_rsi = metrics.get('rsi', 50)
        price_deviation = metrics.get('price_deviation', 0)
        volume_ratio = metrics.get('volume_ratio', 1.0)
        trend_direction = metrics.get('trend_direction', 0)
        
        # Log detailed analysis
        self.logger.info(f"{symbol} [Mean Reversion] - RSI: {current_rsi:.2f}, Mean Dev: {price_deviation:.4f}, Vol Ratio: {volume_ratio:.2f}")
        self.logger.info(f"{symbol} [Mean Reversion] - Signal Strength: Long={long_signal_strength:.4f}, Short={short_signal_strength:.4f}")
        
        # Determine signal threshold
        threshold = self.parameters.get('mean_threshold', 0.005) * 2  # Twice the deviation threshold
        
        # Generate trading signals
        signals = []
        
        # Check for long/buy signal
        if long_signal_strength > threshold:
            # Strong bullish mean reversion signal
            self.logger.info(f"STRONG MEAN REVERSION LONG signal for {symbol} (strength: {long_signal_strength:.4f})")
            
            # Calculate position size
            position_size = self.calculate_position_size({
                'price': current_price,
                'side': 'buy',
                'symbol': symbol,
                'signal_strength': long_signal_strength
            })
            
            if position_size > 0:
                # Set stop loss and take profit levels based on volatility
                std_dev = market_data['close'].rolling(window=self.mean_period).std().iloc[-1] / current_price
                
                # Dynamic stop and take profit levels (proportional to volatility)
                stop_loss_pct = max(0.005, min(0.02, std_dev * 2))  # Min 0.5%, max 2%
                take_profit_pct = max(0.01, min(0.03, std_dev * 4))  # Min 1%, max 3%
                
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
                
                self.logger.info(f"Mean Reversion LONG signal for {symbol}: Price={current_price}, Size={position_size}, "
                                f"SL={stop_loss} ({stop_loss_pct*100:.2f}%), TP={take_profit} ({take_profit_pct*100:.2f}%)")
                
                signals.append({
                    'type': 'MARKET',
                    'side': 'buy',
                    'amount': position_size,
                    'price': None,  # Market order
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'strategy': 'mean_reversion'  # Tag the strategy that generated this signal
                })
        
        # Check for short/sell signal
        elif short_signal_strength > threshold:
            # Strong bearish mean reversion signal
            self.logger.info(f"STRONG MEAN REVERSION SHORT signal for {symbol} (strength: {short_signal_strength:.4f})")
            
            # Calculate position size
            position_size = self.calculate_position_size({
                'price': current_price,
                'side': 'sell',
                'symbol': symbol,
                'signal_strength': short_signal_strength
            })
            
            if position_size > 0:
                # Dynamic stop and take profit based on volatility
                std_dev = market_data['close'].rolling(window=self.mean_period).std().iloc[-1] / current_price
                
                stop_loss_pct = max(0.005, min(0.02, std_dev * 2))
                take_profit_pct = max(0.01, min(0.03, std_dev * 4))
                
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)
                
                self.logger.info(f"Mean Reversion SHORT signal for {symbol}: Price={current_price}, Size={position_size}, "
                                f"SL={stop_loss} ({stop_loss_pct*100:.2f}%), TP={take_profit} ({take_profit_pct*100:.2f}%)")
                
                signals.append({
                    'type': 'MARKET',
                    'side': 'sell',
                    'amount': position_size,
                    'price': None,  # Market order
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'strategy': 'mean_reversion'
                })
        
        else:
            self.logger.info(f"No mean reversion signals for {symbol} - insufficient signal strength")
        
        return signals if signals else None

    def calculate_position_size(self, signal: dict) -> float:
        """Calculate position size with risk management for mean reversion strategy."""
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
        
        # For mean reversion, use slightly smaller positions than trend following
        risk_per_trade *= 0.8
        
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
        
        # Scale with signal strength
        risk_amount = risk_amount * (0.5 + signal_strength)
        
        # Scale position size based on consecutive losses (reduce size after losses)
        if self.consecutive_losses > 0:
            risk_reduction = min(0.5, self.consecutive_losses * 0.2)  # Reduce by up to 50%
            risk_amount *= (1 - risk_reduction)
            self.logger.info(f"Reducing position size by {risk_reduction*100:.0f}% due to {self.consecutive_losses} consecutive losses")
        
        # Dynamic risk based on coin
        if 'BTC' in symbol:
            price_factor = 0.15  # BTC is expensive, use smaller positions
        elif any(coin in symbol for coin in ['ETH', 'AVAX']):
            price_factor = 0.3  # Medium price coins
        else:
            price_factor = 0.7  # Lower price coins
        
        # Base contract calculation
        contract_value = price * 1  # Value of a single contract
        
        # Calculate position size
        position_size = (risk_amount / contract_value) * price_factor
        
        # Scale by leverage
        leverage = 10  # Default leverage
        position_size = position_size * leverage
        
        # Round to appropriate precision
        if price < 1:
            position_size = round(position_size)
        else:
            position_size = round(position_size, 1)
        
        # Ensure minimum viable position and respect maximum
        position_size = max(11, min(position_size, max_position_size))
        
        self.logger.info(f"Mean Reversion calculated position size for {symbol}: {position_size} contracts")
        
        return position_size
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not 0 <= self.oversold <= 100:
            return False
        if not 0 <= self.overbought <= 100:
            return False
        if self.oversold >= self.overbought:
            return False
        if not isinstance(self.rsi_period, int) or self.rsi_period <= 0:
            return False
        if not isinstance(self.mean_period, int) or self.mean_period <= 0:
            return False
        return True 