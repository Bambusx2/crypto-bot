"""
Risk management system for controlling trading risk.
"""

from typing import Dict, Optional
import logging

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
        
        # Position sizing parameters
        self.max_position_size = config['position_sizing']['max_position_size']
        self.risk_per_trade = config['position_sizing']['risk_per_trade']
        
        # Portfolio parameters
        self.max_drawdown = config['portfolio']['max_drawdown']
        self.max_open_positions = config['portfolio']['max_open_positions']
        self.correlation_threshold = config['portfolio']['correlation_threshold']
        
        # Track open positions and portfolio state
        self.open_positions: Dict[str, dict] = {}
        self.current_drawdown = 0.0
        self.peak_value = 0.0
    
    def validate_trade(self, symbol: str, signal: dict) -> bool:
        """
        Validate if a trade meets risk management criteria.
        
        Args:
            symbol: Trading pair symbol
            signal: Trading signal dictionary
            
        Returns:
            True if trade is valid, False otherwise
        """
        # Check if we're already at max positions
        if len(self.open_positions) >= self.max_open_positions:
            self.logger.warning("Maximum number of open positions reached")
            return False
        
        # Get current futures balance
        try:
            futures_balance = self.exchange.client.fetch_balance({'type': 'future'})
            available_balance = float(futures_balance.get('USDT', {}).get('free', 0))
            
            # Calculate required margin with leverage
            price = signal.get('price', 0)
            amount = signal.get('amount', 0)
            leverage = self.config.get('leverage', 10)  # Default to 10x if not specified
            required_margin = (price * amount) / leverage
            
            # Add 5% buffer for fees and price movements
            required_margin *= 1.05
            
            if available_balance < required_margin:
                self.logger.warning(f"Insufficient margin: need ${required_margin:.2f}, have ${available_balance:.2f}")
                return False
        except Exception as e:
            self.logger.error(f"Error checking balance: {str(e)}")
            return False
        
        # Check position size
        position_size = self._calculate_position_value(signal)
        if position_size > self.max_position_size:
            self.logger.warning(f"Position size {position_size} exceeds maximum {self.max_position_size}")
            return False
        
        # Check drawdown
        if self.current_drawdown > self.max_drawdown:
            self.logger.warning(f"Current drawdown {self.current_drawdown} exceeds maximum {self.max_drawdown}")
            return False
        
        # Validate stop loss and take profit
        if not self._validate_risk_reward(signal):
            return False
        
        return True
    
    def _calculate_position_value(self, signal: dict) -> float:
        """
        Calculate the value of a position in terms of portfolio percentage.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Position value as a fraction of portfolio
        """
        return signal['amount'] * signal.get('price', 0)
    
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
        return reward >= risk * 2
    
    def update_position(self, symbol: str, position: Optional[dict] = None):
        """
        Update or remove a position in the tracking system.
        
        Args:
            symbol: Trading pair symbol
            position: Position dictionary or None to remove
        """
        if position is None:
            self.open_positions.pop(symbol, None)
        else:
            self.open_positions[symbol] = position
    
    def update_portfolio_metrics(self, portfolio_value: float):
        """
        Update portfolio metrics including drawdown calculation.
        
        Args:
            portfolio_value: Current portfolio value
        """
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
    
    def get_position_size(self, symbol: str, price: float) -> float:
        """
        Calculate the appropriate position size for a new trade.
        
        Args:
            symbol: Trading pair symbol
            price: Current asset price
            
        Returns:
            Position size in base currency
        """
        # Calculate position size based on risk per trade
        portfolio_value = self._get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade
        
        # Ensure position size doesn't exceed maximum
        max_position_value = portfolio_value * self.max_position_size
        position_size = min(risk_amount / price, max_position_value / price)
        
        return position_size
    
    def _get_portfolio_value(self) -> float:
        """
        Get current portfolio value (placeholder).
        
        Returns:
            Total portfolio value
        """
        # This should be implemented to get actual portfolio value from exchange
        return 10000.0  # Placeholder value 