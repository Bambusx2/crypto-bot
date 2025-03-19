@echo off
echo Setting up data directory and initial trades file...

if not exist data mkdir data
echo [] > data\trades.json

echo Setup complete! 