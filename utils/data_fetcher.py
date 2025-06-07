import yfinance as yf
import pandas as pd
import streamlit as st
import time
import requests
from datetime import datetime

class StockDataFetcher:
    """Class to handle stock data fetching from Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
    
    def fetch_data(self, symbol, start_date, end_date):
        """
        Fetch stock data from Yahoo Finance with retry mechanism
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for data fetching
            end_date (datetime): End date for data fetching
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        # Check cache first
        cache_key = f"{symbol}_{start_date}_{end_date}"
        current_time = time.time()
        
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if current_time - cache_time < self.cache_duration:
                st.info(f"Using cached data for {symbol}")
                return cached_data
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Create ticker object - let yfinance handle its own session
                ticker = yf.Ticker(symbol)
                
                # Add delay between requests to avoid rate limiting
                if attempt > 0:
                    time.sleep(retry_delay * attempt)
                
                # Fetch historical data without timeout parameter
                data = ticker.history(start=start_date, end=end_date)
                
                if data.empty:
                    if attempt < max_retries - 1:
                        st.warning(f"Attempt {attempt + 1} failed for {symbol}. Retrying...")
                        continue
                    else:
                        st.error(f"No data found for symbol '{symbol}' after {max_retries} attempts. Please verify the symbol is correct.")
                        return None
                
                # Reset index to make Date a column
                data.reset_index(inplace=True)
                
                # Cache the successful result
                self.cache[cache_key] = (data.copy(), current_time)
                
                # Ensure we have enough data points
                if len(data) < 100:
                    st.warning(f"Limited data available for {symbol} ({len(data)} points). Consider extending the date range for better predictions.")
                
                return data
                
            except requests.exceptions.RequestException as e:
                if "Too Many Requests" in str(e) or "429" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        st.warning(f"Rate limit hit for {symbol}. Waiting {wait_time} seconds before retry {attempt + 2}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        st.error(f"Rate limit exceeded for {symbol}. Please try again in a few minutes.")
                        return None
                else:
                    st.error(f"Network error fetching data for {symbol}: {str(e)}")
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"Error on attempt {attempt + 1} for {symbol}: {str(e)}. Retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    st.error(f"Failed to fetch data for {symbol} after {max_retries} attempts: {str(e)}")
                    return None
        
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
