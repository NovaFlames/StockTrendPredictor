import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StockVisualizer:
    """Class to handle all visualization tasks for stock prediction app"""
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8'
        }
    
    def plot_historical_data(self, data, symbol):
        """
        Plot historical stock price data with volume
        
        Args:
            data (pd.DataFrame): Stock data
            symbol (str): Stock symbol
            
        Returns:
            plotly.graph_objects.Figure: Historical data plot
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price', 'Volume'),
            row_width=[0.2, 0.7]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add volume bar chart
        fig.add_trace(
            go.Bar(
                x=data['Date'],
                y=data['Volume'],
                name='Volume',
                marker_color=self.colors['info'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Historical Stock Data",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600,
            showlegend=False
        )
        
        # Remove range slider for cleaner look
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        return fig
    
    def plot_predictions_vs_actual(self, actual, predicted, symbol):
        """
        Plot actual vs predicted prices
        
        Args:
            actual (np.array): Actual prices
            predicted (np.array): Predicted prices
            symbol (str): Stock symbol
            
        Returns:
            plotly.graph_objects.Figure: Comparison plot
        """
        # Create date range for x-axis
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=len(actual)),
            periods=len(actual),
            freq='D'
        )
        
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=actual,
                mode='lines',
                name='Actual',
                line=dict(color=self.colors['primary'], width=2)
            )
        )
        
        # Add predicted prices
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predicted,
                mode='lines',
                name='Predicted',
                line=dict(color=self.colors['secondary'], width=2)
            )
        )
        
        # Calculate and display metrics
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        
        fig.update_layout(
            title=f"{symbol} - Actual vs Predicted Prices<br><sub>RMSE: ${rmse:.2f} | MAE: ${mae:.2f}</sub>",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_future_predictions(self, historical_data, predictions, prediction_days, symbol):
        """
        Plot future price predictions with historical context
        
        Args:
            historical_data (pd.DataFrame): Historical stock data
            predictions (np.array): Future price predictions
            prediction_days (int): Number of prediction days
            symbol (str): Stock symbol
            
        Returns:
            plotly.graph_objects.Figure: Future predictions plot
        """
        fig = go.Figure()
        
        # Plot last 30 days of historical data for context
        recent_data = historical_data.tail(30)
        
        fig.add_trace(
            go.Scatter(
                x=recent_data['Date'],
                y=recent_data['Close'],
                mode='lines',
                name='Historical',
                line=dict(color=self.colors['primary'], width=2)
            )
        )
        
        # Create future dates
        last_date = historical_data['Date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=prediction_days,
            freq='D'
        )
        
        # Add predictions
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Predicted',
                line=dict(color=self.colors['secondary'], width=3),
                marker=dict(size=6)
            )
        )
        
        # Add connection line between historical and predicted
        connection_x = [recent_data['Date'].iloc[-1], future_dates[0]]
        connection_y = [recent_data['Close'].iloc[-1], predictions[0]]
        
        fig.add_trace(
            go.Scatter(
                x=connection_x,
                y=connection_y,
                mode='lines',
                name='Connection',
                line=dict(color=self.colors['warning'], width=2, dash='dash'),
                showlegend=False
            )
        )
        
        # Add confidence intervals (simple estimation)
        std_dev = np.std(predictions)
        upper_bound = predictions + std_dev
        lower_bound = predictions - std_dev
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=upper_bound,
                mode='lines',
                name='Upper Bound',
                line=dict(color=self.colors['success'], width=1, dash='dot'),
                opacity=0.5
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=lower_bound,
                mode='lines',
                name='Lower Bound',
                line=dict(color=self.colors['danger'], width=1, dash='dot'),
                fill='tonexty',
                fillcolor=f'rgba(255, 165, 0, 0.1)',
                opacity=0.5
            )
        )
        
        fig.update_layout(
            title=f"{symbol} - Future Price Predictions ({prediction_days} days)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_training_history(self, history):
        """
        Plot model training history
        
        Args:
            history: Keras training history object
            
        Returns:
            plotly.graph_objects.Figure: Training history plot
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Loss', 'Training MAE')
        )
        
        epochs = range(1, len(history.history['loss']) + 1)
        
        # Training and validation loss
        fig.add_trace(
            go.Scatter(
                x=list(epochs),
                y=history.history['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=1
        )
        
        if 'val_loss' in history.history:
            fig.add_trace(
                go.Scatter(
                    x=list(epochs),
                    y=history.history['val_loss'],
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color=self.colors['secondary'])
                ),
                row=1, col=1
            )
        
        # Training and validation MAE
        fig.add_trace(
            go.Scatter(
                x=list(epochs),
                y=history.history['mean_absolute_error'],
                mode='lines',
                name='Training MAE',
                line=dict(color=self.colors['success']),
                showlegend=False
            ),
            row=1, col=2
        )
        
        if 'val_mean_absolute_error' in history.history:
            fig.add_trace(
                go.Scatter(
                    x=list(epochs),
                    y=history.history['val_mean_absolute_error'],
                    mode='lines',
                    name='Validation MAE',
                    line=dict(color=self.colors['danger']),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title="Model Training Progress",
            height=400
        )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=1, col=2)
        
        return fig
    
    def create_correlation_heatmap(self, data):
        """
        Create correlation heatmap for stock features
        
        Args:
            data (pd.DataFrame): Stock data with multiple features
            
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        # Select numeric columns only
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_cols].corr()
        
        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            )
        )
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=500
        )
        
        return fig
