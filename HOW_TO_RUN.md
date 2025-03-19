# How to Run the Trading Bot and Dashboard

## Prerequisites
- Python 3.9+ installed
- Required packages installed via `pip install -r requirements.txt`

## First-Time Setup
Before running the bot for the first time, run the setup script to create the necessary data directories:
```
setup_data.bat
```

## Available Startup Options

### Expert Trading Bot (Recommended)
This is the optimized professional-grade trading bot with multi-factor analysis:
```
start_expert_trading.bat
```
This will start both the expert trading bot and dashboard in separate windows.

### Standard Trading Bot (Multi-Coin Mode)
This is the default trading bot that monitors multiple cryptocurrencies and trades when it finds strong signals:
```
start_both_multi.bat
```
This will start both the trading bot and dashboard in separate windows.

### Fast Trading (Aggressive Settings)
Use this option if you want the bot to generate trades more quickly with smaller SMA windows:
```
start_both_now.bat
```
This uses smaller SMA windows (3 and 5 periods) to generate trades faster.

### Manual/Emergency Trade Execution
For direct trade execution that bypasses the bot's strategy:

```
execute_custom_trade.bat "BTC/USDT" "sell" 11 10
```
Parameters: Symbol, Side (buy/sell), Amount, Leverage

## Starting Components Individually

### Trading Bot Only
```
start_expert_bot.bat     # Expert trading bot
start_bot.bat            # Standard multi-coin mode
```

### Dashboard Only
```
start_dashboard.bat
```

## Stopping the Bot and Dashboard
To stop all running bots and dashboards at once:
```
stop_all.bat
```

## Using the Dashboard

The dashboard provides a user-friendly interface to:
- Monitor trades and performance
- View profit/loss metrics
- Start/stop the trading bot
- Clear trade history
- Edit configuration settings

## Troubleshooting

### Bot Won't Start from Dashboard
If you can't start the bot from the dashboard:
1. Make sure no other bot instances are running
2. Try running one of the bot startup scripts directly
3. Check the console output for any error messages

### Dashboard Not Loading Data
Make sure the data directory exists and contains a valid trades.json file:
```
setup_data.bat
```

### Bot Not Starting
Check that your .env file contains valid API credentials for your exchange. 