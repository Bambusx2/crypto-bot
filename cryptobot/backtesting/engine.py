"""
Backtesting engine for testing trading strategies.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

from ..exchanges.base import BaseExchange
from ..strategies.base import BaseStrategy
from ..risk.manager import RiskManager

class BacktestingEngine:
    """Engine for backtesting trading strategies."""
    
    def __init__(
        self,
        exchange: BaseExchange,
        strategy: BaseStrategy,
        risk_manager: RiskManager,
        initial_balance: float = 10000.0,
        commission: float = 0.001  # 0.1% commission
    ):
        """
        Initialize backtesting engine.
        
        Args:
            exchange: Exchange instance
            strategy: Strategy instance
            risk_manager: Risk manager instance
            initial_balance: Initial account balance
            commission: Trading commission rate
        """
        self.exchange = exchange
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Performance tracking
        self.balance = initial_balance
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.equity_curve: List[float] = [initial_balance]
    
    def run(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1h'
    ) -> dict:
        """
        Run backtest over specified period.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Data timeframe
            
        Returns:
            Dict containing backtest results
        """
        # Fetch historical data
        market_data = self.exchange.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=1000  # Adjust based on date range
        )
        
        # Filter data for backtest period
        market_data = market_data[start_date:end_date]
        
        # Run simulation
        for timestamp, candle in market_data.iterrows():
            # Generate signals
            signals = self.strategy.generate_signals(
                market_data.loc[:timestamp]
            )
            
            if signals:
                for signal in signals:
                    if self.risk_manager.validate_trade(symbol, signal):
                        self._execute_trade(timestamp, symbol, signal, candle)
            
            # Update positions and equity
            self._update_positions(timestamp, candle)
            self.equity_curve.append(self._calculate_equity(candle))
        
        # Calculate performance metrics
        return self._calculate_performance_metrics()
    
    def _execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        signal: dict,
        candle: pd.Series
    ):
        """
        Execute a simulated trade.
        
        Args:
            timestamp: Current timestamp
            symbol: Trading pair symbol
            signal: Trading signal
            candle: Current price candle
        """
        price = signal.get('price', candle['close'])
        amount = signal['amount']
        commission = price * amount * self.commission
        
        if signal['side'] == 'buy':
            cost = price * amount + commission
            if cost <= self.balance:
                self.balance -= cost
                self.positions[symbol] = {
                    'amount': amount,
                    'entry_price': price,
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit')
                }
                
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': 'buy',
                    'price': price,
                    'amount': amount,
                    'commission': commission
                })
        
        elif signal['side'] == 'sell':
            if symbol in self.positions:
                position = self.positions[symbol]
                proceeds = price * position['amount'] - commission
                self.balance += proceeds
                
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': 'sell',
                    'price': price,
                    'amount': position['amount'],
                    'commission': commission,
                    'profit': proceeds - (position['entry_price'] * position['amount'])
                })
                
                del self.positions[symbol]
    
    def _update_positions(self, timestamp: datetime, candle: pd.Series):
        """
        Update open positions and check for stop loss/take profit.
        
        Args:
            timestamp: Current timestamp
            candle: Current price candle
        """
        for symbol, position in list(self.positions.items()):
            current_price = candle['close']
            
            # Check stop loss
            if position['stop_loss'] and current_price <= position['stop_loss']:
                self._execute_trade(
                    timestamp,
                    symbol,
                    {
                        'side': 'sell',
                        'amount': position['amount'],
                        'price': position['stop_loss']
                    },
                    candle
                )
            
            # Check take profit
            elif position['take_profit'] and current_price >= position['take_profit']:
                self._execute_trade(
                    timestamp,
                    symbol,
                    {
                        'side': 'sell',
                        'amount': position['amount'],
                        'price': position['take_profit']
                    },
                    candle
                )
    
    def _calculate_equity(self, candle: pd.Series) -> float:
        """
        Calculate current equity including open positions.
        
        Args:
            candle: Current price candle
            
        Returns:
            Total equity value
        """
        equity = self.balance
        
        for symbol, position in self.positions.items():
            equity += position['amount'] * candle['close']
        
        return equity
    
    def _calculate_performance_metrics(self) -> dict:
        """
        Calculate trading performance metrics.
        
        Returns:
            Dict containing performance metrics
        """
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        winning_trades = [t for t in self.trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('profit', 0) <= 0]
        
        metrics = {
            'initial_balance': self.initial_balance,
            'final_balance': self.equity_curve[-1],
            'total_return': (self.equity_curve[-1] - self.initial_balance) / self.initial_balance,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            'equity_curve': self.equity_curve
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Returns:
            Maximum drawdown as a percentage
        """
        peak = self.equity_curve[0]
        max_drawdown = 0
        
        for value in self.equity_curve[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.plot(self.equity_curve, label='Equity')
        plt.title('Backtest Results')
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close() 