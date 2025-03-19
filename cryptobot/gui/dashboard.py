"""
Trading bot dashboard for monitoring trades and performance.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path
import time
import subprocess
import os
import sys
import psutil
import ccxt

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_trades():
    """Load trades from the trades log file."""
    trades_file = Path("data/trades.json")
    if trades_file.exists():
        try:
            with open(trades_file, "r") as f:
                trades = json.load(f)
                
            # Clean up trades data
            valid_trades = []
            current_time = datetime.utcnow()  # Use UTC time for consistency
            
            for trade in trades:
                try:
                    # Parse entry time
                    try:
                        entry_time = datetime.fromisoformat(trade["entry_time"].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        # If entry time is invalid, skip this trade
                        st.warning(f"Invalid entry time for trade: {trade}")
                        continue
                    
                    # Skip trades with future entry times
                    if entry_time > current_time:
                        st.warning(f"Skipping trade with future entry time: {entry_time}")
                        continue
                    
                    # Handle exit time
                    exit_time = None
                    if trade.get("exit_time"):
                        try:
                            exit_time = datetime.fromisoformat(trade["exit_time"].replace('Z', '+00:00'))
                            # Skip trades with future exit times
                            if exit_time > current_time:
                                exit_time = None
                        except (ValueError, AttributeError):
                            exit_time = None
                    
                    # Determine if trade is active
                    is_active = (
                        exit_time is None or
                        float(trade.get("exit_price", 0)) == 0 or
                        float(trade.get("realized_pnl", 0)) == 0
                    )
                    
                    # Update trade data
                    trade_data = {
                        "entry_time": entry_time.isoformat(),
                        "exit_time": exit_time.isoformat() if exit_time else current_time.isoformat(),
                        "symbol": trade.get("symbol", "UNKNOWN"),
                        "position_side": trade.get("position_side", "unknown"),
                        "entry_price": float(trade.get("entry_price", 0)),
                        "exit_price": float(trade.get("exit_price", 0)),
                        "amount": float(trade.get("amount", 0)),
                        "realized_pnl": float(trade.get("realized_pnl", 0)),
                        "is_active": is_active
                    }
                    
                    valid_trades.append(trade_data)
                    
                except (ValueError, KeyError) as e:
                    st.warning(f"Skipping invalid trade: {str(e)}")
                    continue
            
            return valid_trades
            
        except (json.JSONDecodeError, IOError) as e:
            st.error(f"Error loading trades: {str(e)}")
    return []

def calculate_unrealized_pnl(trade, current_price):
    """Calculate unrealized PnL for an active trade."""
    if not current_price:
        return 0.0
        
    entry_price = float(trade['entry_price'])
    amount = float(trade['amount'])
    
    if trade['position_side'].lower() == 'long':
        return (current_price - entry_price) * amount
    else:  # short position
        return (entry_price - current_price) * amount

def get_current_prices(symbols):
    """Get current prices for multiple symbols."""
    try:
        # Initialize Binance Futures client
        client = ccxt.binance({
            'options': {'defaultType': 'future'},
            'enableRateLimit': True
        })
        
        prices = {}
        for symbol in symbols:
            try:
                ticker = client.fetch_ticker(symbol)
                prices[symbol] = float(ticker['last'])
            except Exception as e:
                st.warning(f"Could not fetch price for {symbol}: {str(e)}")
                prices[symbol] = None
                
        return prices
    except Exception as e:
        st.error(f"Error fetching prices: {str(e)}")
        return {}

def calculate_metrics(trades):
    """Calculate trading metrics."""
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_profit": 0,
            "unrealized_profit": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "average_profit": 0,
            "average_loss": 0,
            "active_trades": 0
        }
    
    # Separate completed and active trades
    completed_trades = [t for t in trades if not t.get("is_active", False)]
    active_trades = [t for t in trades if t.get("is_active", False)]
    
    # Get current prices for active trades
    symbols = list(set(t['symbol'] for t in active_trades))
    current_prices = get_current_prices(symbols)
    
    # Calculate unrealized PnL for active trades
    unrealized_pnl = sum(
        calculate_unrealized_pnl(trade, current_prices.get(trade['symbol']))
        for trade in active_trades
    )
    
    # Calculate metrics from completed trades
    total_trades = len(completed_trades)
    winning_trades = len([t for t in completed_trades if t["realized_pnl"] > 0])
    losing_trades = len([t for t in completed_trades if t["realized_pnl"] < 0])
    
    profits = [t["realized_pnl"] for t in completed_trades if t["realized_pnl"] > 0]
    losses = [abs(t["realized_pnl"]) for t in completed_trades if t["realized_pnl"] < 0]
    
    total_profit = sum(profits) if profits else 0
    total_loss = sum(losses) if losses else 0
    net_profit = total_profit - total_loss
    
    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        "total_profit": net_profit,
        "unrealized_profit": unrealized_pnl,
        "largest_win": max(profits) if profits else 0,
        "largest_loss": max(losses) if losses else 0,
        "average_profit": sum(profits) / len(profits) if profits else 0,
        "average_loss": sum(losses) / len(losses) if losses else 0,
        "active_trades": len(active_trades)
    }

def plot_equity_curve(trades):
    """Plot equity curve using plotly."""
    if not trades:
        return go.Figure()
    
    # Sort trades by entry time
    trades = sorted(trades, key=lambda x: x["entry_time"])
    
    # Calculate cumulative PnL only from completed trades
    cumulative_pnl = []
    running_total = 0
    timestamps = []
    
    for trade in trades:
        if not trade.get("is_active", False):  # Only include completed trades
            running_total += trade["realized_pnl"]
            cumulative_pnl.append(running_total)
            timestamps.append(trade["exit_time"])
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=cumulative_pnl,
            mode='lines',
            name='Equity Curve',
            line=dict(color='green' if running_total >= 0 else 'red', width=2)
        )
    )
    
    fig.update_layout(
        title='Equity Curve (Completed Trades)',
        xaxis_title='Time',
        yaxis_title='Cumulative Profit/Loss (USDT)',
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )
    
    return fig

def plot_trade_distribution(trades):
    """Plot trade profit/loss distribution."""
    if not trades:
        return go.Figure()
    
    pnls = [t["realized_pnl"] for t in trades]
    
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=pnls,
            nbinsx=20,
            name='Trade Distribution',
            marker_color=['green' if x >= 0 else 'red' for x in pnls]
        )
    )
    
    fig.update_layout(
        title='Trade Profit/Loss Distribution',
        xaxis_title='Profit/Loss (USDT)',
        yaxis_title='Number of Trades',
        bargap=0.1
    )
    
    return fig

def add_sidebar_controls():
    """Add control buttons to the sidebar."""
    st.sidebar.title("Bot Controls")
    
    # Check if bot is running
    bot_running = check_if_bot_running()
    
    status_text = "Running" if bot_running else "Stopped"
    status_color = "green" if bot_running else "red"
    st.sidebar.markdown(f"<h3>Status: <span style='color:{status_color};'>{status_text}</span></h3>", unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    
    if not bot_running:
        if col1.button("Start Bot", type="primary"):
            try:
                # Get the absolute path to the project root
                root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                
                # Create a unique identifier for the bot process
                pid_file = os.path.join(root_dir, "bot_process.pid")
                
                # Start the bot in a new process with shell=True to ensure it works in Windows
                process = subprocess.Popen(
                    ["python", "-m", "cryptobot", "--config", "config/default_config.yml"],
                    cwd=root_dir,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    shell=True,
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
                )
                
                # Save the process ID to a file for later reference
                with open(pid_file, "w") as f:
                    f.write(str(process.pid))
                
                st.sidebar.success("Bot started! Check the console window for output.")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to start bot: {str(e)}")
    else:
        if col1.button("Stop Bot", type="primary"):
            try:
                # Get the absolute path to the project root
                root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                pid_file = os.path.join(root_dir, "bot_process.pid")
                
                # Check if we have a stored process ID
                if os.path.exists(pid_file):
                    with open(pid_file, "r") as f:
                        pid = int(f.read().strip())
                    
                    # Kill the specific process
                    if sys.platform == "win32":
                        subprocess.call(["taskkill", "/F", "/PID", str(pid), "/T"], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        subprocess.call(["kill", str(pid)], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Remove the PID file
                    os.remove(pid_file)
                else:
                    # Fallback to the old method if no PID file exists
                    process_name = "cryptobot"
                    if sys.platform == "win32":
                        # For Windows, find and kill only the cryptobot process
                        subprocess.call(["taskkill", "/F", "/FI", f"WINDOWTITLE eq *{process_name}*"], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        subprocess.call(["pkill", "-f", process_name], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                st.sidebar.success("Bot stopped!")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to stop bot: {str(e)}")
    
    if col2.button("Clear Trades"):
        try:
            # Clear the trades file
            trades_file = Path("data/trades.json")
            if trades_file.exists():
                with open(trades_file, "w") as f:
                    json.dump([], f)
                st.sidebar.success("Trades cleared!")
                time.sleep(1)
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to clear trades: {str(e)}")
    
    # Add configuration editor
    st.sidebar.subheader("Configuration")
    config_editor = st.sidebar.expander("Edit Config", expanded=False)
    
    with config_editor:
        config_path = "config/default_config.yml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_text = f.read()
            
            new_config = st.text_area("Edit Configuration", config_text, height=300)
            
            if st.button("Save Config"):
                try:
                    with open(config_path, "w") as f:
                        f.write(new_config)
                    st.success("Configuration saved!")
                except Exception as e:
                    st.error(f"Failed to save configuration: {str(e)}")

def check_if_bot_running():
    """Check if the trading bot is currently running."""
    try:
        # Get the absolute path to the project root
        root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        pid_file = os.path.join(root_dir, "bot_process.pid")
        
        # First check if we have a PID file
        if os.path.exists(pid_file):
            with open(pid_file, "r") as f:
                pid_content = f.read().strip()
            
            # Check if the PID file contains a placeholder ("starting") or a valid PID
            if pid_content.isdigit():
                pid = int(pid_content)
                
                # Check if the process is still running
                if sys.platform == "win32":
                    try:
                        # This will raise an exception if the process is not running
                        process = psutil.Process(pid)
                        if process.is_running():
                            # Check the command line to make sure it's our bot
                            cmd_line = " ".join(process.cmdline()).lower()
                            if "python" in cmd_line and ("cryptobot" in cmd_line or "bot.py" in cmd_line):
                                return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
                        # Process doesn't exist or we can't access it
                        st.sidebar.warning(f"PID file exists but process check failed: {str(e)}")
                        # Clean up the stale PID file
                        try:
                            os.remove(pid_file)
                        except:
                            pass
                        return False
                else:
                    # For Linux/Mac, we can use the os.kill with signal 0 to check if process exists
                    try:
                        os.kill(pid, 0)
                        return True
                    except OSError:
                        # Clean up the stale PID file
                        try:
                            os.remove(pid_file)
                        except:
                            pass
                        return False
            else:
                # PID file contains a placeholder like "starting"
                # Check if there are any recent bot processes
                st.sidebar.info(f"Bot is in startup process: '{pid_content}'")
                
                # Look for recent bot activity
                log_file = os.path.join(root_dir, "logs", "trading.log")
                if os.path.exists(log_file):
                    # If log was modified in the last 30 seconds, consider bot running
                    if time.time() - os.path.getmtime(log_file) < 30:
                        return True
                
                # Check for cryptobot processes
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmd_line = " ".join(proc.cmdline()).lower() if proc.cmdline() else ""
                        if "python" in proc.name().lower() and ("cryptobot" in cmd_line or "bot.py" in cmd_line):
                            # Found a running bot process - update PID file with actual PID
                            with open(pid_file, "w") as f:
                                f.write(str(proc.pid))
                            return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                        continue
                
                # If PID file exists but no bot is running, clean up the stale PID file
                try:
                    os.remove(pid_file)
                except:
                    pass
                return False
                
        # Fallback to checking all running processes for cryptobot
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmd_line = " ".join(proc.cmdline()).lower() if proc.cmdline() else ""
                if "python" in proc.name().lower() and ("cryptobot" in cmd_line or "bot.py" in cmd_line):
                    # Found a running bot process - create PID file
                    with open(pid_file, "w") as f:
                        f.write(str(proc.pid))
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                continue
                
        # Check log file for recent activity as last resort
        log_file = os.path.join(root_dir, "logs", "trading.log")
        if os.path.exists(log_file):
            # Check if log was modified in the last 30 seconds
            if time.time() - os.path.getmtime(log_file) < 30:
                return True
                
        return False
    except Exception as e:
        st.sidebar.error(f"Error checking if bot is running: {str(e)}")
        return False

def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Trading Bot Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Trading Bot Dashboard")
    
    # Load and process trades
    trades = load_trades()
    metrics = calculate_metrics(trades)
    
    # Display key metrics in a clean grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_pnl = metrics['total_profit'] + metrics['unrealized_profit']
        profit_color = "green" if total_pnl >= 0 else "red"
        st.markdown(
            f"### Total Profit/Loss\n"
            f"<h2 style='color: {profit_color}'>${total_pnl:.2f}</h2>\n"
            f"<small>Realized: ${metrics['total_profit']:.2f} | "
            f"Unrealized: ${metrics['unrealized_profit']:.2f}</small>",
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(f"### Win Rate\n<h2>{metrics['win_rate']:.1f}%</h2>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"### Completed Trades\n<h2>{metrics['total_trades']}</h2>", unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"### Active Trades\n<h2>{metrics['active_trades']}</h2>", unsafe_allow_html=True)
    
    # Additional metrics in expandable section
    with st.expander("Detailed Statistics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Winning Trades", metrics['winning_trades'])
            st.metric("Largest Win", f"${metrics['largest_win']:.2f}")
        with col2:
            st.metric("Losing Trades", metrics['losing_trades'])
            st.metric("Largest Loss", f"${metrics['largest_loss']:.2f}")
        with col3:
            st.metric("Average Win", f"${metrics['average_profit']:.2f}")
            st.metric("Win/Loss Ratio", f"{metrics['win_rate'] / (100 - metrics['win_rate']):.2f}" if metrics['win_rate'] < 100 else "∞")
        with col4:
            st.metric("Average Loss", f"${metrics['average_loss']:.2f}")
            st.metric("Total Volume", f"${sum(t['amount'] * t['entry_price'] for t in trades):.2f}")
    
    # Plot equity curve
    st.plotly_chart(plot_equity_curve(trades), use_container_width=True)
    
    # Display trades tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Active Trades")
        active_trades = [t for t in trades if t.get('is_active', False)]
        if active_trades:
            df_active = pd.DataFrame(active_trades)
            df_active['entry_time'] = pd.to_datetime(df_active['entry_time'])
            
            # Get current prices and calculate unrealized PnL
            symbols = list(set(df_active['symbol']))
            current_prices = get_current_prices(symbols)
            
            # Add current price and unrealized PnL columns
            df_active['current_price'] = df_active['symbol'].map(current_prices)
            df_active['unrealized_pnl'] = df_active.apply(
                lambda x: calculate_unrealized_pnl(x, x['current_price']), axis=1
            )
            
            # Format and display active trades
            display_cols = ['entry_time', 'symbol', 'position_side', 'entry_price', 'current_price', 'unrealized_pnl', 'amount']
            df_active = df_active[display_cols].sort_values('entry_time', ascending=False)
            
            # Format the display
            df_active['entry_time'] = df_active['entry_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_active['entry_price'] = df_active['entry_price'].apply(lambda x: f"${x:.4f}")
            df_active['current_price'] = df_active['current_price'].apply(lambda x: f"${x:.4f}" if x else "N/A")
            df_active['unrealized_pnl'] = df_active['unrealized_pnl'].apply(
                lambda x: f"🟢 ${x:.2f}" if x > 0 else f"🔴 ${x:.2f}" if x < 0 else f"${x:.2f}"
            )
            df_active['amount'] = df_active['amount'].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(df_active, use_container_width=True)
        else:
            st.info("No active trades")
    
    with col2:
        st.subheader("Recent Completed Trades")
        completed_trades = [t for t in trades if not t.get('is_active', False)]
        if completed_trades:
            df_completed = pd.DataFrame(completed_trades)
            df_completed['entry_time'] = pd.to_datetime(df_completed['entry_time'])
            df_completed['exit_time'] = pd.to_datetime(df_completed['exit_time'])
            
            # Calculate duration
            df_completed['duration'] = (df_completed['exit_time'] - df_completed['entry_time']).apply(
                lambda x: f"{x.total_seconds() / 3600:.1f}h"
            )
            
            # Format and display completed trades
            display_cols = ['exit_time', 'symbol', 'position_side', 'realized_pnl', 'duration']
            df_completed = df_completed[display_cols].sort_values('exit_time', ascending=False).head(10)
            
            # Format the display
            df_completed['exit_time'] = df_completed['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_completed['realized_pnl'] = df_completed['realized_pnl'].apply(
                lambda x: f"🟢 ${x:.2f}" if x > 0 else f"🔴 ${x:.2f}"
            )
            
            st.dataframe(df_completed, use_container_width=True)
        else:
            st.info("No completed trades")

if __name__ == "__main__":
    main() 