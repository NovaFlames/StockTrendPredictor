import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.data_fetcher import StockDataFetcher
from utils.preprocessor import DataPreprocessor
from utils.lstm_model import LSTMModel
from utils.visualizer import StockVisualizer

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction with LSTM",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📈 Stock Price Prediction using LSTM")
    st.markdown("This application uses LSTM neural networks to predict future stock prices based on historical data.")
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("Configuration")
        
        # Stock symbol input
        stock_symbol = st.text_input(
            "Enter Stock Symbol",
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
        ).upper()
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365*2),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # Model parameters
        st.subheader("Model Parameters")
        sequence_length = st.slider("Sequence Length", 10, 100, 60, help="Number of days to look back for prediction")
        prediction_days = st.slider("Prediction Days", 1, 30, 7, help="Number of days to predict into the future")
        epochs = st.slider("Training Epochs", 10, 200, 50, help="Number of training iterations")
        
        # Train model button
        train_model = st.button("🚀 Train Model & Predict", type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if train_model and stock_symbol:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Fetch data
                status_text.text("📊 Fetching stock data...")
                progress_bar.progress(10)
                
                data_fetcher = StockDataFetcher()
                stock_data = data_fetcher.fetch_data(stock_symbol, start_date, end_date)
                
                if stock_data is None or stock_data.empty:
                    st.error(f"❌ Could not fetch data for symbol '{stock_symbol}'. Please check if the symbol is valid.")
                    return
                
                # Step 2: Preprocess data
                status_text.text("🔧 Preprocessing data...")
                progress_bar.progress(25)
                
                preprocessor = DataPreprocessor(sequence_length)
                X_train, y_train, X_test, y_test, scaler = preprocessor.prepare_data(stock_data)
                
                # Step 3: Train model
                status_text.text("🤖 Training LSTM model...")
                progress_bar.progress(40)
                
                lstm_model = LSTMModel(sequence_length)
                model, history = lstm_model.build_and_train(X_train, y_train, X_test, y_test, epochs)
                
                # Step 4: Make predictions
                status_text.text("🔮 Making predictions...")
                progress_bar.progress(70)
                
                predictions = lstm_model.predict(model, X_test, scaler)
                future_predictions = lstm_model.predict_future(model, stock_data, scaler, prediction_days, sequence_length)
                
                # Step 5: Visualize results
                status_text.text("📊 Creating visualizations...")
                progress_bar.progress(90)
                
                visualizer = StockVisualizer()
                
                # Historical data chart
                st.subheader("📊 Historical Stock Price")
                fig_history = visualizer.plot_historical_data(stock_data, stock_symbol)
                st.plotly_chart(fig_history, use_container_width=True)
                
                # Predictions vs actual chart
                st.subheader("🎯 Model Performance (Test Data)")
                actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                fig_performance = visualizer.plot_predictions_vs_actual(actual_prices, predictions, stock_symbol)
                st.plotly_chart(fig_performance, use_container_width=True)
                
                # Future predictions chart
                st.subheader("🔮 Future Price Predictions")
                fig_future = visualizer.plot_future_predictions(stock_data, future_predictions, prediction_days, stock_symbol)
                st.plotly_chart(fig_future, use_container_width=True)
                
                # Training history
                st.subheader("📈 Training Progress")
                fig_training = visualizer.plot_training_history(history)
                st.plotly_chart(fig_training, use_container_width=True)
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                # Display success message
                st.success(f"🎉 Successfully trained LSTM model for {stock_symbol}!")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.error("Please check your inputs and try again.")
    
    with col2:
        if train_model and stock_symbol:
            try:
                # Display model metrics and statistics
                st.subheader("📊 Model Statistics")
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                
                mse = mean_squared_error(actual_prices, predictions)
                mae = mean_absolute_error(actual_prices, predictions)
                rmse = np.sqrt(mse)
                
                # Display metrics in a nice format
                st.metric("Root Mean Square Error", f"${rmse:.2f}")
                st.metric("Mean Absolute Error", f"${mae:.2f}")
                
                # Display stock info
                st.subheader("📋 Stock Information")
                latest_price = stock_data['Close'].iloc[-1]
                price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
                price_change_pct = (price_change / stock_data['Close'].iloc[-2]) * 100
                
                st.metric(
                    "Latest Price", 
                    f"${latest_price:.2f}",
                    delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                )
                
                # Future prediction summary
                if 'future_predictions' in locals():
                    st.subheader("🔮 Prediction Summary")
                    future_avg = np.mean(future_predictions)
                    current_price = stock_data['Close'].iloc[-1]
                    predicted_change = future_avg - current_price
                    predicted_change_pct = (predicted_change / current_price) * 100
                    
                    st.metric(
                        f"Avg. Price ({prediction_days} days)",
                        f"${future_avg:.2f}",
                        delta=f"{predicted_change:+.2f} ({predicted_change_pct:+.2f}%)"
                    )
                    
                    # Show prediction confidence
                    confidence = max(0, min(100, 100 - (rmse / latest_price * 100)))
                    st.metric("Model Confidence", f"{confidence:.1f}%")
                
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")
        
        else:
            st.info("👈 Configure parameters and click 'Train Model & Predict' to start the analysis.")
            
            # Display sample information
            st.subheader("ℹ️ How it works")
            st.write("""
            1. **Data Fetching**: Downloads historical stock data
            2. **Preprocessing**: Normalizes and structures data for LSTM
            3. **Model Training**: Trains LSTM neural network
            4. **Prediction**: Generates future price predictions
            5. **Visualization**: Shows results with interactive charts
            """)
            
            st.subheader("💡 Tips")
            st.write("""
            - Use popular stock symbols (AAPL, GOOGL, MSFT, etc.)
            - Longer sequence lengths capture more patterns
            - More epochs improve accuracy but take longer
            - 2+ years of data recommended for better predictions
            """)

if __name__ == "__main__":
    main()
