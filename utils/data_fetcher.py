import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime

class StockDataFetcher:
    """Class to handle stock data fetching from Yahoo Finance"""
    
    def __init__(self):
        pass
    
    def fetch_data(self, symbol, start_date, end_date):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for data fetching
            end_date (datetime): End date for data fetching
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                st.error(f"No data found for symbol '{symbol}'. Please check if the symbol is correct.")
                return None
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Ensure we have enough data points
            if len(data) < 100:
                st.warning(f"Limited data available for {symbol}. Consider extending the date range for better predictions.")
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_stock_info(self, symbol):
        """
        Get additional stock information
        
        Args:
            symbol (str): Stock ticker symbol
            
        Returns:
            dict: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            return {'name': symbol, 'sector': 'N/A', 'industry': 'N/A', 'market_cap': 'N/A', 'currency': 'USD'}
    
    def validate_symbol(self, symbol):
        """
        Validate if a stock symbol exists
        
        Args:
            symbol (str): Stock ticker symbol
            
        Returns:
            bool: True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            return not data.empty
        except:
            return False
