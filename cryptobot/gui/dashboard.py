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

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_trades():
    """Load trades from the trades log file."""
    trades_file = Path("data/trades.json")
    if trades_file.exists():
        try:
            with open(trades_file, "r") as f:
                trades = json.load(f)
                
                # Clean up trades data - ensure all required fields are present
                valid_trades = []
                for trade in trades:
                    # Ensure all necessary fields exist
                    if all(field in trade for field in ["entry_time", "exit_time", "position_side", "entry_price", "exit_price", "realized_pnl"]):
                        # Make sure numeric fields are floats
                        for field in ["entry_price", "exit_price", "realized_pnl"]:
                            trade[field] = float(trade[field])
                        
                        # Add profit field if not present
                        if "profit" not in trade:
                            trade["profit"] = float(trade["realized_pnl"])
                            
                        valid_trades.append(trade)
                
                return valid_trades
        except (json.JSONDecodeError, IOError) as e:
            st.error(f"Error loading trades: {str(e)}")
    return []

def calculate_metrics(trades):
    """Calculate trading metrics."""
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_profit": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "average_profit": 0,
            "average_loss": 0
        }
    
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t["realized_pnl"] > 0])
    losing_trades = len([t for t in trades if t["realized_pnl"] < 0])
    
    profits = [t["realized_pnl"] for t in trades if t["realized_pnl"] > 0]
    losses = [t["realized_pnl"] for t in trades if t["realized_pnl"] < 0]
    
    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        "total_profit": sum(t["realized_pnl"] for t in trades),
        "largest_win": max(profits) if profits else 0,
        "largest_loss": min(losses) if losses else 0,
        "average_profit": sum(profits) / len(profits) if profits else 0,
        "average_loss": sum(losses) / len(losses) if losses else 0
    }

def plot_equity_curve(trades):
    """Plot equity curve using plotly."""
    if not trades:
        return go.Figure()
    
    cumulative_pnl = []
    running_total = 0
    timestamps = []
    
    for trade in trades:
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
            line=dict(color='green' if running_total >= 0 else 'red')
        )
    )
    
    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Time',
        yaxis_title='Cumulative Profit/Loss (USDT)',
        hovermode='x unified'
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
    st.set_page_config(
        page_title="Trading Bot Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Trading Bot Dashboard")
    
    # Add sidebar controls
    add_sidebar_controls()
    
    # Add auto-refresh checkbox
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=True)
    
    # Add refresh button
    if st.sidebar.button("Manual Refresh"):
        st.rerun()
    
    # Add data file info
    trades_file = Path("data/trades.json")
    if trades_file.exists():
        modified_time = datetime.fromtimestamp(trades_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        st.sidebar.info(f"Data last updated: {modified_time}")
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Load latest trades
        trades = load_trades()
        
        if not trades:
            st.warning("No trade data found. The bot may not have executed any trades yet.")
            return
            
        metrics = calculate_metrics(trades)
        
        # Display metrics
        with col1:
            st.metric("Total Profit/Loss", f"${metrics['total_profit']:.2f}")
        with col2:
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        with col3:
            st.metric("Total Trades", metrics['total_trades'])
        with col4:
            st.metric("Average Profit", f"${metrics['average_profit']:.2f}")
        
        # Equity curve with unique key
        st.plotly_chart(plot_equity_curve(trades), use_container_width=True, key="equity_curve")
        
        # Recent trades table
        st.subheader("Recent Trades")
        if trades:
            df = pd.DataFrame(trades[-10:])  # Show last 10 trades
            
            # Format the DataFrame for display
            if 'profit' in df.columns:
                df['profit_indicator'] = df['profit'].apply(lambda x: '🟢' if x > 0 else ('⚪' if x == 0 else '🔴'))
            else:
                df['profit_indicator'] = df['realized_pnl'].apply(lambda x: '🟢' if x > 0 else ('⚪' if x == 0 else '🔴'))
            
            # Add symbol column if not present
            if 'symbol' not in df.columns:
                df['symbol'] = 'Unknown'
                
            # Format prices to be more readable
            if 'entry_price' in df.columns:
                df['entry_price'] = df['entry_price'].apply(lambda x: f"${x:.4f}")
            if 'exit_price' in df.columns:
                df['exit_price'] = df['exit_price'].apply(lambda x: f"${x:.4f}")
            if 'realized_pnl' in df.columns:
                df['realized_pnl'] = df['realized_pnl'].apply(lambda x: f"${x:.2f}")
            
            # Select and order columns
            display_columns = ['entry_time', 'exit_time', 'symbol', 'position_side', 'entry_price', 
                              'exit_price', 'realized_pnl', 'profit_indicator']
            
            # Make sure all columns exist
            display_columns = [col for col in display_columns if col in df.columns]
            
            # Display the table
            st.table(df[display_columns])
        else:
            st.info("No trades yet")
        
        # Trading metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Winning Trades")
            st.write(f"Count: {metrics['winning_trades']}")
            st.write(f"Largest Win: ${metrics['largest_win']:.2f}")
            st.write(f"Average Win: ${metrics['average_profit']:.2f}")
        
        with col2:
            st.subheader("Losing Trades")
            st.write(f"Count: {metrics['losing_trades']}")
            st.write(f"Largest Loss: ${abs(metrics['largest_loss']):.2f}")
            st.write(f"Average Loss: ${abs(metrics['average_loss']):.2f}")
    
    except Exception as e:
        st.error(f"Error displaying dashboard: {str(e)}")
        st.exception(e)
        
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main() 