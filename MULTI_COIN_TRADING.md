# Multi-Coin Trading Bot

## Overview

The trading bot has been updated to monitor multiple cryptocurrencies simultaneously and only open positions when strong trading signals are detected. This approach offers several advantages:

1. **Diversification** - By monitoring multiple coins, the bot can find the best trading opportunities across the market
2. **Reduced Risk** - The bot will only trade when it identifies a high-confidence setup
3. **Better Performance** - Trading only strong signals improves the win rate

## Coins Being Monitored

The bot is now tracking the following cryptocurrencies:

- BTC (Bitcoin)
- ETH (Ethereum)
- XRP (Ripple)
- ADA (Cardano)
- LINK (Chainlink)
- HBAR (Hedera)
- AVAX (Avalanche)
- DOT (Polkadot)
- NEAR (Near Protocol)
- ICP (Internet Computer)
- ONDO (Ondo Finance)
- RENDER (Render Network)
- S (SingularityNET)
- FET (Fetch.ai)
- SEI (Sei Network)
- STX (Stacks)
- IOTA (IOTA)
- KDA (Kadena)
- FLUX (Flux)

## Trading Strategy Improvements

The trading strategy has been enhanced with additional filters to ensure only high-quality trades are taken:

1. **Volume Confirmation** - Trades are only executed when volume is above average
2. **Price Volatility Analysis** - The bot measures recent price volatility to avoid choppy markets
3. **Strong SMA Crossovers** - Only trades when moving averages show a clear trend
4. **Symbol-Specific Position Sizing** - The bot adjusts position size based on the specific cryptocurrency

## How to Use

1. **Run the Multi-Coin Trading Bot**:
   ```
   start_both_multi.bat
   ```
   This will start both the trading bot and the dashboard.

2. **Monitor the Dashboard**:
   The dashboard will show all open positions and recent trades across all coins.

3. **Check Logs**:
   For detailed information about trading signals and decisions, check the log files in the `logs` directory.

## Risk Management

The bot includes several risk management features:

- Maximum of 3 open positions at once
- Each position is sized appropriately for the specific cryptocurrency
- Stop-loss and take-profit levels are set for each trade
- Maximum drawdown protection
- Trailing stops to lock in profits

## Performance Tracking

All trades are recorded in the `data/trades.json` file, which tracks:

- Entry and exit prices
- Position size
- Profit/loss
- Duration of the trade
- The cryptocurrency that was traded

You can review this data to evaluate the bot's performance over time. 