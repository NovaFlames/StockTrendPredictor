import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import streamlit as st

class LSTMModel:
    """Class to handle LSTM model creation, training, and prediction"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            tensorflow.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            # Second LSTM layer with return sequences
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer without return sequences
            LSTM(units=50),
            Dropout(0.2),
            
            # Dense output layer
            Dense(units=1)
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        return model
    
    def build_and_train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """
        Build and train the LSTM model
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (trained_model, training_history)
        """
        try:
            # Reshape input data for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Build model
            self.model = self.build_model((X_train.shape[1], 1))
            
            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
            
            # Create progress placeholder
            progress_placeholder = st.empty()
            
            # Custom callback to update Streamlit progress
            class StreamlitProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_placeholder.progress(int(progress * 30) + 40)  # 40-70% of total progress
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, reduce_lr, StreamlitProgressCallback()],
                verbose=0
            )
            
            return self.model, history
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            raise e
    
    def predict(self, model, X_test, scaler):
        """
        Make predictions using trained model
        
        Args:
            model: Trained LSTM model
            X_test: Test data
            scaler: Data scaler for inverse transformation
            
        Returns:
            np.array: Predictions in original scale
        """
        try:
            # Reshape test data
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Make predictions
            predictions = model.predict(X_test, verbose=0)
            
            # Inverse transform predictions
            predictions = scaler.inverse_transform(predictions).flatten()
            
            return predictions
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            raise e
    
    def predict_future(self, model, data, scaler, days=7, sequence_length=60):
        """
        Predict future stock prices
        
        Args:
            model: Trained LSTM model
            data: Historical stock data
            scaler: Data scaler
            days (int): Number of days to predict
            sequence_length (int): Length of input sequence
            
        Returns:
            np.array: Future price predictions
        """
        try:
            # Get the last sequence_length prices
            last_sequence = data['Close'].values[-sequence_length:].reshape(-1, 1)
            
            # Scale the data
            last_sequence_scaled = scaler.transform(last_sequence)
            
            # Initialize predictions list
            future_predictions = []
            
            # Current sequence for prediction
            current_sequence = last_sequence_scaled.copy()
            
            # Predict future prices iteratively
            for _ in range(days):
                # Reshape for prediction
                X_pred = current_sequence.reshape(1, sequence_length, 1)
                
                # Make prediction
                next_pred = model.predict(X_pred, verbose=0)
                
                # Store prediction
                future_predictions.append(next_pred[0, 0])
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_pred[0, 0]
            
            # Convert predictions back to original scale
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = scaler.inverse_transform(future_predictions).flatten()
            
            return future_predictions
            
        except Exception as e:
            st.error(f"Error predicting future prices: {str(e)}")
            raise e
    
    def get_model_summary(self):
        """
        Get model architecture summary
        
        Returns:
            str: Model summary as string
        """
        if self.model is None:
            return "Model not built yet."
        
        import io
        summary_string = io.StringIO()
        self.model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
        return summary_string.getvalue()
