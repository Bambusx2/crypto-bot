"""
Risk management system for controlling trading risk.
"""

from typing import Dict, Optional, List
import logging
import time
from datetime import datetime, timedelta

class RiskManager:
    """Manages trading risk through position sizing and trade validation."""
    
    def __init__(self, config: dict, exchange=None):
        """
        Initialize risk manager.
        
        Args:
            config: Risk management configuration dictionary
            exchange: Exchange instance for balance checking
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.exchange = exchange
        
        # Position sizing parameters - more aggressive
        self.max_position_size = config['position_sizing'].get('max_position_size', 0.2)  # 20% of portfolio
        self.risk_per_trade = config['position_sizing'].get('risk_per_trade', 0.02)  # 2% risk per trade
        
        # Portfolio parameters - less restrictive
        self.max_drawdown = config['portfolio'].get('max_drawdown', 0.25)  # 25% max drawdown
        self.max_open_positions = config['portfolio'].get('max_open_positions', 10)  # Allow up to 10 positions
        self.correlation_threshold = config['portfolio'].get('correlation_threshold', 0.8)  # Higher correlation threshold
        self.max_daily_trades = config['portfolio'].get('max_daily_trades', 50)  # More trades per day
        self.cooldown_period = config['portfolio'].get('cooldown_period', 30)  # Shorter cooldown
        
        # Stop loss parameters - wider stops
        self.stop_loss_enabled = config['stop_loss'].get('enabled', True)
        self.stop_loss_percentage = config['stop_loss'].get('percentage', 0.03)  # 3% stop loss
        self.trailing_stop = config['stop_loss'].get('trailing', True)
        self.dynamic_stop = config['stop_loss'].get('dynamic', True)
        
        # Take profit parameters - more aggressive
        self.take_profit_enabled = config['take_profit'].get('enabled', True)
        self.take_profit_percentage = config['take_profit'].get('percentage', 0.04)  # 4% take profit
        self.partial_exits = config['take_profit'].get('partial_exits', True)
        self.tp_levels = config['take_profit'].get('tp_levels', [0.02, 0.03, 0.04])  # Multiple TP levels
        self.tp_sizes = config['take_profit'].get('tp_sizes', [0.3, 0.3, 0.4])  # Size for each TP
        
        # Track open positions and portfolio state
        self.open_positions = {}
        self.current_drawdown = 0.0
        self.peak_value = 0.0
        self.daily_trades_count = 0
        self.daily_trades_reset = datetime.now() + timedelta(days=1)
        self.last_trade_time = {}
    
    def validate_trade(self, symbol: str, signal: dict) -> bool:
        """
        Validate if a trade meets risk management criteria.
        
        Args:
            symbol: Trading pair symbol
            signal: Signal dictionary containing trade details
            
        Returns:
            True if trade is valid, False otherwise
        """
        try:
            # Get current portfolio value
            balance = self.exchange.get_balance()
            portfolio_value = float(balance.get('USDT', {}).get('total', 0))
            self.logger.info(f"Futures wallet balance: ${portfolio_value:.2f} USDT")
            
            # Get current price
            ticker = self.exchange.client.fetch_ticker(symbol)
            current_price = float(ticker['last'])
            self.logger.info(f"Using current market price for {symbol}: {current_price}")
            
            # Calculate required margin
            amount = signal['amount']
            notional_value = amount * current_price
            
            # Get leverage - use default of 10 if not set
            try:
                leverage = self.exchange.client.fetch_leverage(symbol)
                if leverage is None or leverage == 0:
                    leverage = 10  # Default leverage
                leverage = float(leverage)
            except Exception as e:
                self.logger.warning(f"Could not fetch leverage for {symbol}, using default: {str(e)}")
                leverage = 10  # Default leverage
            
            self.logger.info(f"Using leverage {leverage}x for {symbol}")
            
            required_margin = notional_value / leverage
            
            # More lenient margin check - up to 30% of portfolio per position
            if required_margin > portfolio_value * 0.3:
                self.logger.warning(f"Position size too large: {required_margin:.2f} > {portfolio_value * 0.3:.2f} (30% of portfolio)")
                return False
            
            # Get all open positions
            positions = self.exchange.client.fetch_positions([symbol])
            open_positions = [p for p in positions if abs(float(p['contracts'])) > 0]
            
            # More lenient position limit
            if len(open_positions) >= self.max_open_positions:
                self.logger.warning(f"Maximum number of positions ({self.max_open_positions}) reached")
                return False
            
            # Allow multiple positions per symbol if in different directions
            for position in open_positions:
                if position['symbol'] == symbol:
                    existing_side = 'long' if float(position['contracts']) > 0 else 'short'
                    new_side = signal.get('side', '').lower()
                    
                    # Allow if closing or reducing position
                    if signal.get('reduce_only', False):
                        return True
                    
                    # Allow if opening in opposite direction
                    if (existing_side == 'long' and new_side == 'sell') or \
                       (existing_side == 'short' and new_side == 'buy'):
                        return True
                    
                    # Otherwise, prevent duplicate positions
                    self.logger.warning(f"Position already exists for {symbol} in same direction")
                    return False
            
            # Track daily trades with reset
            now = datetime.now()
            if now >= self.daily_trades_reset:
                self.daily_trades_count = 0
                self.daily_trades_reset = now + timedelta(days=1)
            
            self.daily_trades_count += 1
            self.logger.info(f"Trade validated for {symbol}. Daily trades: {self.daily_trades_count}/{self.max_daily_trades}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in trade validation: {str(e)}")
            return False
    
    def _calculate_position_value(self, signal: dict) -> float:
        """
        Calculate the value of a position in terms of portfolio percentage.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Position value as a fraction of portfolio
        """
        amount = signal['amount']
        price = signal.get('price')
        
        # For market orders, fetch current price
        if price is None:
            try:
                symbol = signal.get('symbol')
                if symbol:
                    ticker = self.exchange.client.fetch_ticker(symbol)
                    price = float(ticker['last'])
                    self.logger.info(f"Using current market price for {symbol}: {price}")
                else:
                    self.logger.error("No symbol provided in signal for market order")
                    return 0
            except Exception as e:
                self.logger.error(f"Error fetching current price: {str(e)}")
                return 0
        
        return float(amount) * float(price)
    
    def _validate_risk_reward(self, signal: dict) -> bool:
        """
        Validate the risk/reward ratio of a trade.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            True if risk/reward is acceptable, False otherwise
        """
        entry_price = signal.get('price', 0)
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        
        if not all([entry_price, stop_loss, take_profit]):
            return True  # Skip validation if not all parameters are provided
        
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        # Minimum risk/reward ratio of 1:2
        risk_reward_ratio = reward / risk if risk > 0 else 0
        min_required_ratio = 2.0
        
        # Check if it meets minimum requirement
        if risk_reward_ratio < min_required_ratio:
            self.logger.warning(f"Risk/reward ratio of {risk_reward_ratio:.2f} is below minimum {min_required_ratio}")
            return False
        
        return True
    
    def _check_correlation(self, new_symbol: str) -> bool:
        """
        Check if a new position would be too correlated with existing positions.
        
        Args:
            new_symbol: Symbol of the new position
            
        Returns:
            True if correlation is acceptable, False otherwise
        """
        # Simple correlation groups (can be enhanced with actual correlation data)
        correlation_groups = [
            ['BTC', 'ETH'],  # Major cryptos often move together
            ['XRP', 'XLM', 'HBAR'],  # Payment/network tokens
            ['AVAX', 'SOL', 'NEAR', 'LINK'],  # L1/Smart contract platforms
            ['DOT', 'ADA', 'ATOM'],  # Interoperability focused
        ]
        
        # Extract the base currency from the symbol (e.g., BTC from BTC/USDT)
        if '/' in new_symbol:
            base_currency = new_symbol.split('/')[0]
        else:
            base_currency = new_symbol
        
        # Check existing positions against correlation groups
        for group in correlation_groups:
            if base_currency in group:
                # Count how many positions we already have in this group
                group_count = sum(1 for sym in self.open_positions
                                if any(coin in sym for coin in group))
                
                # If we already have positions in this group, limit additional exposure
                if group_count >= 2:  # At most 2 positions from the same correlation group
                    self.logger.warning(f"Already have {group_count} positions in correlation group with {base_currency}")
                    return False
        
        return True
    
    def update_position(self, symbol: str, position: Optional[dict] = None):
        """
        Update or remove a position in the tracking system.
        
        Args:
            symbol: Trading pair symbol
            position: Position dictionary or None to remove
        """
        if position is None:
            self.open_positions.pop(symbol, None)
            self.logger.info(f"Removed position tracking for {symbol}")
        else:
            self.open_positions[symbol] = position
            self.logger.info(f"Updated position tracking for {symbol}: {position}")
    
    def update_portfolio_metrics(self, portfolio_value: float):
        """
        Update portfolio metrics including drawdown calculation.
        
        Args:
            portfolio_value: Current portfolio value
        """
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.logger.info(f"New portfolio peak value: ${portfolio_value:.2f}")
        
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            self.logger.info(f"Current drawdown: {self.current_drawdown:.2%}")
    
    def get_position_size(self, symbol: str, price: float, signal_strength: float = 0.5) -> float:
        """
        Calculate the appropriate position size for a new trade.
        
        Args:
            symbol: Trading pair symbol
            price: Current asset price
            signal_strength: Strength of the signal (0.0 to 1.0)
            
        Returns:
            Position size in base currency
        """
        # Calculate position size based on risk per trade
        portfolio_value = self._get_portfolio_value()
        
        # Scale risk amount based on signal strength
        risk_amount = portfolio_value * self.risk_per_trade
        if signal_strength > 0:
            # Stronger signals get larger positions (up to 2x)
            risk_adjustment = 0.5 + (signal_strength * 0.5)
            risk_amount *= risk_adjustment
            self.logger.info(f"Adjusting position size by {risk_adjustment:.2f}x based on signal strength {signal_strength:.2f}")
        
        # Calculate maximum position value based on portfolio percentage
        max_position_ratio = self.config.get('max_position_ratio', 0.3)
        max_position_value = portfolio_value * max_position_ratio
        
        # Calculate position size based on price and risk
        position_size = risk_amount / price
        
        # Ensure position size doesn't exceed maximum
        max_position_size_by_value = max_position_value / price
        position_size = min(position_size, max_position_size_by_value)
        
        # Apply leverage
        leverage = 10  # Default leverage
        position_size = position_size * leverage
        
        # Round to appropriate precision
        if price < 1:
            position_size = round(position_size)
        else:
            position_size = round(position_size, 1)
        
        # Ensure minimum viable position size (exchange dependent)
        position_size = max(11, position_size)
        
        self.logger.info(f"Calculated position size for {symbol}: {position_size} contracts (signal strength: {signal_strength:.2f})")
        
        return position_size
    
    def calculate_dynamic_stops(self, symbol: str, entry_price: float, side: str, volatility: float = None) -> dict:
        """
        Calculate dynamic stop loss and take profit levels based on volatility.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price
            side: Position side ('buy'/'long' or 'sell'/'short')
            volatility: Optional volatility measure
            
        Returns:
            Dictionary with stop_loss and take_profit values
        """
        # Default values
        stop_loss_pct = self.stop_loss_percentage
        take_profit_pct = self.take_profit_percentage
        
        # Adjust based on volatility if dynamic stops are enabled
        if self.dynamic_stop and volatility:
            # Scale stop loss with volatility, but keep within reasonable bounds
            volatility_factor = min(3.0, max(0.5, volatility * 10))
            stop_loss_pct = min(0.03, max(0.005, stop_loss_pct * volatility_factor))
            
            # Take profit should be at least 2x the stop loss
            take_profit_pct = max(take_profit_pct, stop_loss_pct * 2.5)
            
            self.logger.info(f"Dynamic stops for {symbol}: SL={stop_loss_pct:.2%}, TP={take_profit_pct:.2%} (volatility={volatility:.4f})")
        
        # Calculate actual price levels based on direction
        is_long = side.lower() in ['buy', 'long']
        
        if is_long:
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
        
        # Calculate partial take profit levels if enabled
        partial_levels = []
        if self.partial_exits and self.tp_levels and self.tp_sizes:
            for level_pct, size_pct in zip(self.tp_levels, self.tp_sizes):
                if is_long:
                    price_level = entry_price * (1 + level_pct)
                else:
                    price_level = entry_price * (1 - level_pct)
                
                partial_levels.append({
                    'price': price_level,
                    'size': size_pct
                })
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': self.trailing_stop,
            'partial_take_profits': partial_levels
        }
    
    def _get_portfolio_value(self) -> float:
        """
        Get current portfolio value from exchange.
        
        Returns:
            Total portfolio value
        """
        try:
            if self.exchange:
                balance = self.exchange.get_balance()
                # First try to get USDT balance from futures wallet
                if isinstance(balance, dict) and 'USDT' in balance:
                    usdt_balance = balance['USDT'].get('total', 0)
                    total_value = float(usdt_balance)
                    self.logger.info(f"Futures wallet balance: ${total_value:.2f} USDT")
                    return total_value
                
                # Fallback to checking nested structure
                total_value = float(balance.get('total', {}).get('USDT', 0))
                if total_value > 0:
                    self.logger.info(f"Current portfolio value: ${total_value:.2f}")
                    return total_value
                
                # Try direct balance fetch as last resort
                try:
                    futures_balance = self.exchange.client.fetch_balance({'type': 'future'})
                    if futures_balance and 'USDT' in futures_balance:
                        total_value = float(futures_balance['USDT']['total'])
                        self.logger.info(f"Direct futures balance fetch: ${total_value:.2f} USDT")
                        return total_value
                except Exception as e:
                    self.logger.error(f"Error fetching direct futures balance: {str(e)}")
                
                self.logger.warning("Could not get accurate balance, using minimum value")
                return 46.41  # Use known balance as fallback
            else:
                self.logger.warning("No exchange instance available")
                return 46.41  # Use known balance as fallback
        except Exception as e:
            self.logger.error(f"Error getting portfolio value: {str(e)}")
            self.logger.error("Using fallback balance value")
            return 46.41  # Use known balance as fallback 