"""
Trading Strategy Backtester - Web Application

This web application implements a moving average crossover strategy backtester, with a modern web interface.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import random
import math
import json
import plotly
import plotly.graph_objs as go
from datetime import datetime, timedelta
import plotly.express as px

app = Flask(__name__)

def generate_price_data(start_price=100, days=100, volatility=0.01, upward_drift=0.0001):
    """
    Generate simulated daily stock prices with random walk and slight upward drift.
    
    Args:
        start_price: Initial price of the stock
        days: Number of days to simulate
        volatility: Daily price volatility (standard deviation)
        upward_drift: Slight upward bias in price movement
        
    Returns:
        List of simulated daily prices
    """
    prices = [start_price]
    current_price = start_price

    for _ in range(days-1):
        daily_return = random.gauss(upward_drift, volatility)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    return prices