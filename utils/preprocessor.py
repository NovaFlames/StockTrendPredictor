import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st

class DataPreprocessor:
    """Class to handle data preprocessing for LSTM model"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def prepare_data(self, data, target_column='Close', test_size=0.2):
        """
        Prepare data for LSTM training
        
        Args:
            data (pd.DataFrame): Stock data
            target_column (str): Column to predict
            test_size (float): Proportion of data for testing
            
        Returns:
            tuple: X_train, y_train, X_test, y_test, scaler
        """
        try:
            # Extract the target column
            prices = data[target_column].values.reshape(-1, 1)
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(prices)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            
            if len(X) == 0:
                raise ValueError("Not enough data to create sequences. Try reducing sequence length or increasing date range.")
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            return X_train, y_train, X_test, y_test, self.scaler
            
        except Exception as e:
            st.error(f"Error in data preprocessing: {str(e)}")
            raise e
    
    def _create_sequences(self, data):
        """
        Create sequences for LSTM training
        
        Args:
            data (np.array): Scaled price data
            
        Returns:
            tuple: X (sequences), y (targets)
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def prepare_prediction_data(self, data, target_column='Close'):
        """
        Prepare data for making predictions
        
        Args:
            data (pd.DataFrame): Stock data
            target_column (str): Column to predict
            
        Returns:
            np.array: Scaled and sequenced data ready for prediction
        """
        try:
            # Get the last sequence_length points
            prices = data[target_column].values[-self.sequence_length:].reshape(-1, 1)
            
            # Scale the data using the existing scaler
            scaled_data = self.scaler.transform(prices)
            
            # Reshape for LSTM input
            prediction_data = scaled_data.reshape(1, self.sequence_length, 1)
            
            return prediction_data
            
        except Exception as e:
            st.error(f"Error preparing prediction data: {str(e)}")
            raise e
    
    def inverse_transform(self, scaled_predictions):
        """
        Convert scaled predictions back to original scale
        
        Args:
            scaled_predictions (np.array): Scaled prediction values
            
        Returns:
            np.array: Predictions in original scale
        """
        return self.scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()
    
    def add_technical_indicators(self, data):
        """
        Add technical indicators to the dataset
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        try:
            # Simple Moving Averages
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            data['MA_10'] = data['Close'].rolling(window=10).mean()
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            
            # Relative Strength Index (RSI)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['BB_middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
            data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
            
            # Volume indicators
            data['Volume_MA'] = data['Volume'].rolling(window=10).mean()
            
            return data
            
        except Exception as e:
            st.warning(f"Could not add technical indicators: {str(e)}")
            return data
