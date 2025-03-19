"""
Trading strategies package.
"""

from .base import BaseStrategy
from .trend_following import Strategy

__all__ = ['BaseStrategy', 'Strategy'] 