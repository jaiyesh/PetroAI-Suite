import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title=" PetroForecast AI",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5ea;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .performance-metrics {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

class OilProductionForecaster:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_scalers = {}
        self.training_history = None
        self.forecast_mode = "univariate"  # or "multivariate"
        
    def prepare_data(self, df, target_col, feature_cols, lookback_days, forecast_mode="univariate"):
        """Prepare data for LSTM training with different forecasting modes"""
        self.forecast_mode = forecast_mode
        
        if forecast_mode == "univariate":
            # Use only oil production for forecasting (more stable)
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(df[[target_col]])
            
            # Create sequences for univariate prediction
            X, y = [], []
            for i in range(lookback_days, len(scaled_data)):
                X.append(scaled_data[i-lookback_days:i, 0])
                y.append(scaled_data[i, 0])
            
            return np.array(X).reshape(-1, lookback_days, 1), np.array(y)
        
        else:
            # Multivariate approach (original)
            self.scaler = MinMaxScaler()
            scaled_target = self.scaler.fit_transform(df[[target_col]])
            
            scaled_features = []
            for col in feature_cols:
                if col != target_col:
                    scaler = MinMaxScaler()
                    self.feature_scalers[col] = scaler
                    scaled_col = scaler.fit_transform(df[[col]])
                    scaled_features.append(scaled_col)
            
            # Combine all features
            if scaled_features:
                X_data = np.concatenate([scaled_target] + scaled_features, axis=1)
            else:
                X_data = scaled_target
            
            # Create sequences
            X, y = [], []
            for i in range(lookback_days, len(X_data)):
                X.append(X_data[i-lookback_days:i])
                y.append(scaled_target[i, 0])  # Only predict oil production
            
            return np.array(X), np.array(y)
    
    def build_model(self, input_shape, lstm_units=50, learning_rate=0.001):
        """Build LSTM model optimized for the forecasting mode"""
        if self.forecast_mode == "univariate":
            # Simpler model for univariate forecasting
            model = Sequential([
                LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(lstm_units//2, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
        else:
            # More complex model for multivariate
            model = Sequential([
                LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(lstm_units//2, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dropout(0.1),
                Dense(1)
            ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=False,
            verbose=0,
            callbacks=callbacks
        )
        
        self.training_history = history
        return history
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X, verbose=0)
        return self.scaler.inverse_transform(predictions)
    
    def forecast_future_univariate(self, last_sequence, days_ahead):
        """Generate future forecasts using univariate approach (more stable)"""
        forecast = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            # Predict next value
            pred = self.model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
            
            # Store forecast (denormalized)
            forecast_value = self.scaler.inverse_transform(pred.reshape(-1, 1))[0, 0]
            forecast.append(max(0, forecast_value))
            
            # Update sequence (add prediction, remove oldest)
            current_sequence = np.append(current_sequence[1:], pred[0, 0])
        
        return np.array(forecast)
    
    def forecast_future_multivariate(self, last_sequence, days_ahead, feature_cols, df):
        """Generate future forecasts with proper feature handling"""
        forecast = []
        current_sequence = last_sequence.copy()
        
        # Calculate feature trends from recent data
        recent_data = df[feature_cols].tail(60).ffill()
        feature_trends = {}
        
        for col in feature_cols[1:]:  # Skip oil production
            if col in recent_data.columns and len(recent_data[col].dropna()) > 10:
                # Calculate trend using linear regression
                values = recent_data[col].dropna().values
                x = np.arange(len(values))
                if len(values) > 1:
                    trend = np.polyfit(x, values, 1)[0]  # Linear trend
                    feature_trends[col] = trend
                else:
                    feature_trends[col] = 0
            else:
                feature_trends[col] = 0
        
        for day in range(days_ahead):
            # Predict next oil production value
            pred = self.model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
            
            # Store forecast (denormalized oil production only)
            forecast_value = self.scaler.inverse_transform(pred.reshape(-1, 1))[0, 0]
            forecast.append(max(0, forecast_value))
            
            # Create next input sequence with realistic feature values
            next_features = np.zeros((1, len(feature_cols)))
            next_features[0, 0] = pred[0, 0]  # Predicted oil production (normalized)
            
            # For other features, use trend-based forecasting
            for i, col in enumerate(feature_cols[1:], 1):
                if col in self.feature_scalers and col in feature_trends:
                    # Get last known actual value
                    last_actual = recent_data[col].iloc[-1]
                    
                    # Project forward using trend with some dampening
                    dampening = 0.95 ** day  # Reduce trend impact over time
                    projected_value = last_actual + (feature_trends[col] * day * dampening)
                    
                    # Add some realistic noise
                    noise = np.random.normal(0, recent_data[col].std() * 0.02)
                    projected_value += noise
                    
                    # Normalize using the appropriate scaler
                    scaler = self.feature_scalers[col]
                    normalized_value = scaler.transform([[projected_value]])[0, 0]
                    next_features[0, i] = np.clip(normalized_value, 0, 1)
                else:
                    # Use last known value with slight decay
                    decay_factor = 0.999 ** day
                    next_features[0, i] = current_sequence[-1, i] * decay_factor
            
            # Update sequence (remove oldest, add newest)
            current_sequence = np.vstack([current_sequence[1:], next_features])
        
        return np.array(forecast)
    
    def forecast_future(self, last_sequence, days_ahead, feature_cols=None, df=None):
        """Main forecasting method that chooses the appropriate approach"""
        if self.forecast_mode == "univariate":
            return self.forecast_future_univariate(last_sequence, days_ahead)
        else:
            return self.forecast_future_multivariate(last_sequence, days_ahead, feature_cols, df)

def calculate_metrics(actual, predicted):
    """Calculate performance metrics with proper error handling"""
    # Ensure arrays are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Remove any NaN or infinite values
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
    
    # Basic metrics
    mse = mean_squared_error(actual_clean, predicted_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_clean, predicted_clean)
    
    # MAPE calculation with proper handling of zero values
    # Method 1: Exclude zero values
    non_zero_mask = actual_clean != 0
    if np.sum(non_zero_mask) > 0:
        mape_values = np.abs((actual_clean[non_zero_mask] - predicted_clean[non_zero_mask]) / actual_clean[non_zero_mask])
        mape = np.mean(mape_values) * 100
    else:
        # If all actuals are zero, use symmetric MAPE
        mape = np.mean(np.abs(actual_clean - predicted_clean) / (np.abs(actual_clean) + np.abs(predicted_clean) + 1e-8)) * 100
    
    # Additional metrics for petroleum engineering
    # R-squared
    ss_res = np.sum((actual_clean - predicted_clean) ** 2)
    ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Mean Absolute Scaled Error (MASE) - useful for production data
    if len(actual_clean) > 1:
        naive_forecast_error = np.mean(np.abs(np.diff(actual_clean)))
        mase = mae / naive_forecast_error if naive_forecast_error != 0 else np.nan
    else:
        mase = np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'MASE': mase,
        'Data_Points': len(actual_clean)
    }

# Main Streamlit App
def main():
    st.title("🛢️ PetroForecast AI")
    st.markdown("### Smart Oil Production Forecasting Dashboard")
    
    # Sidebar for parameters
    st.sidebar.header("🔧 Model Configuration")
    
    # Forecasting mode selection
    forecast_mode = st.sidebar.selectbox(
        "Forecasting Mode",
        ["univariate", "multivariate"],
        help="Univariate: Uses only oil production history (more stable). Multivariate: Uses multiple features (more complex)."
    )
    
    # Model parameters
    lookback_days = st.sidebar.slider("Lookback Window (days)", 7, 180, 60)
    forecast_days = st.sidebar.slider("Forecast Period (days)", 1, 180, 30)
    epochs = st.sidebar.slider("Training Epochs", 10, 200, 50)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.0001, 0.001, 0.01], index=1)
    lstm_units = st.sidebar.slider("LSTM Units", 10, 200, 50)
    
    # Feature selection
    st.sidebar.header("📊 Feature Selection")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Production Data CSV", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            
            # Data preprocessing
            df['DATEPRD'] = pd.to_datetime(df['DATEPRD'], format='%d-%b-%y')
            df = df.sort_values('DATEPRD').reset_index(drop=True)
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📅 Total Records", len(df))
            with col2:
                st.metric("🗓️ Date Range", f"{(df['DATEPRD'].max() - df['DATEPRD'].min()).days} days")
            with col3:
                production_days = len(df[df['BORE_OIL_VOL'] > 0])
                st.metric("⚡ Production Days", f"{production_days} ({production_days/len(df)*100:.1f}%)")
            with col4:
                st.metric("📈 Avg Production", f"{df['BORE_OIL_VOL'].mean():.0f} bbl/day")
            
            # Feature selection
            available_features = [
                'BORE_OIL_VOL',
                'ON_STREAM_HRS',
                'AVG_DOWNHOLE_PRESSURE',
                'AVG_DOWNHOLE_TEMPERATURE',
                'AVG_WHP_P',
                'AVG_CHOKE_SIZE_P',
                'Total Water Injection'
            ]
            
            # Filter available features based on what's in the dataset
            available_features = [f for f in available_features if f in df.columns]
            
            selected_features = st.sidebar.multiselect(
                "Select Features",
                available_features,
                default=['BORE_OIL_VOL', 'AVG_DOWNHOLE_PRESSURE', 'AVG_WHP_P', 'Total Water Injection']
            )
            
            if 'BORE_OIL_VOL' not in selected_features:
                selected_features = ['BORE_OIL_VOL'] + selected_features
            
            # Data visualization
            st.header("📊 Data Exploration")
            
            # Production time series
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['DATEPRD'],
                y=df['BORE_OIL_VOL'],
                mode='lines',
                name='Oil Production',
                line=dict(color='#667eea', width=2)
            ))
            fig.update_layout(
                title="Historical Oil Production",
                xaxis_title="Date",
                yaxis_title="Production (bbl/day)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlation
            if len(selected_features) > 1:
                corr_data = df[selected_features].corr()
                fig_corr = px.imshow(
                    corr_data,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Model training section
            st.header("🤖 Model Training & Forecasting")
            
            if st.button("🚀 Train LSTM Model & Generate Forecast", type="primary"):
                with st.spinner("Training LSTM model... This may take a few minutes."):
                    try:
                        # Prepare data
                        forecaster = OilProductionForecaster()
                        
                        # Filter out rows with missing values and zero production for training
                        # Keep some zero values for realistic modeling, but ensure we have enough non-zero data
                        production_data = df[df['BORE_OIL_VOL'] > 0].copy()  # Non-zero production days
                        zero_production = df[df['BORE_OIL_VOL'] == 0].copy()  # Zero production days
                        
                        # If we have both types, combine them with more weight on production days
                        if len(production_data) > lookback_days + 50 and len(zero_production) > 0:
                            # Use 90% production days, 10% zero days for realistic modeling
                            n_zeros = min(len(zero_production), len(production_data) // 9)
                            zero_sample = zero_production.sample(n=n_zeros, random_state=42) if n_zeros > 0 else pd.DataFrame()
                            clean_df = pd.concat([production_data, zero_sample]).sort_values('DATEPRD').reset_index(drop=True)
                        else:
                            # Use only production days if not enough data
                            clean_df = production_data.copy()
                        
                        # Final check for selected features
                        clean_df = clean_df[selected_features].dropna()
                        
                        if len(clean_df) < lookback_days + 50:
                            st.error(f"❌ Insufficient data after cleaning. Need at least {lookback_days + 50} records, got {len(clean_df)}.")
                            st.info("💡 Try reducing the lookback window or check your data quality.")
                            return
                        
                        # Prepare training data
                        X, y = forecaster.prepare_data(clean_df, 'BORE_OIL_VOL', selected_features, lookback_days, forecast_mode)
                        
                        # Split data
                        split_idx = int(len(X) * 0.8)
                        X_train, X_test = X[:split_idx], X[split_idx:]
                        y_train, y_test = y[:split_idx], y[split_idx:]
                        
                        # Train model
                        history = forecaster.train_model(X_train, y_train, epochs=epochs)
                        
                        # Make predictions
                        train_pred = forecaster.predict(X_train)
                        test_pred = forecaster.predict(X_test)
                        
                        # Calculate metrics
                        train_metrics = calculate_metrics(
                            forecaster.scaler.inverse_transform(y_train.reshape(-1, 1)).flatten(),
                            train_pred.flatten()
                        )
                        test_metrics = calculate_metrics(
                            forecaster.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
                            test_pred.flatten()
                        )
                        
                        # Generate future forecast
                        if forecast_mode == "univariate":
                            last_sequence = X[-1].flatten()  # For univariate, flatten to 1D
                            future_forecast = forecaster.forecast_future(last_sequence, forecast_days)
                        else:
                            last_sequence = X[-1]
                            future_forecast = forecaster.forecast_future(last_sequence, forecast_days, selected_features, clean_df)
                        
                        # Store additional info in session state
                        st.session_state.forecaster = forecaster
                        st.session_state.forecast = future_forecast
                        st.session_state.test_metrics = test_metrics
                        st.session_state.history = history
                        st.session_state.df = df
                        st.session_state.clean_df = clean_df
                        st.session_state.test_pred = test_pred
                        st.session_state.y_test = y_test
                        st.session_state.forecast_mode = forecast_mode
                        st.session_state.X_train = X_train  # Store training data
                        st.session_state.y_train = y_train
                        st.session_state.lookback_days = lookback_days
                        
                        # Success message with performance assessment
                        if test_metrics['RMSE'] < clean_df['BORE_OIL_VOL'].mean() * 0.2:
                            st.success(f"✅ Model trained successfully! Good performance: RMSE = {test_metrics['RMSE']:.1f} bbl/day")
                        else:
                            st.warning(f"⚠️ Model trained but performance could be improved: RMSE = {test_metrics['RMSE']:.1f} bbl/day. Try adjusting parameters.")
                        
                        # Additional insights
                        if test_metrics.get('MAPE', float('inf')) > 50:
                            st.info("💡 High MAPE detected. This often occurs with low production values or many zero days. Consider using RMSE and MAE for evaluation.")
                        
                        if test_metrics.get('R2', 0) < 0.5:
                            st.info("💡 Low R² suggests the model is struggling to capture patterns. Try: 1) Increasing lookback window, 2) Using univariate mode, 3) More training epochs.")
                        
                        st.success("✅ Model training completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
                        st.exception(e)
            
            # Display results if model is trained
            if 'forecaster' in st.session_state:
                st.header("📈 Results & Analysis")
                
                # Performance metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                metrics = st.session_state.test_metrics
                
                with col1:
                    st.metric("RMSE", f"{metrics['RMSE']:.1f} bbl/day")
                with col2:
                    st.metric("MAE", f"{metrics['MAE']:.1f} bbl/day")
                with col3:
                    if 'R2' in metrics:
                        st.metric("R²", f"{metrics['R2']:.3f}")
                    else:
                        st.metric("R²", "N/A")
                with col4:
                    st.metric("Avg Forecast", f"{np.mean(st.session_state.forecast):.0f} bbl/day")
                
                # Calculate actual and predicted test values first (before they are used)
                actual_test = st.session_state.forecaster.scaler.inverse_transform(
                    st.session_state.y_test.reshape(-1, 1)
                ).flatten()
                predicted_test = st.session_state.test_pred.flatten()
                
                # Forecast visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Historical + Forecast chart with proper dates
                    fig = go.Figure()
                    
                    # Show more historical context (last 6 months or 180 days)
                    context_days = min(180, len(st.session_state.clean_df))
                    hist_data = st.session_state.clean_df.tail(context_days)
                    
                    # Historical data
                    if 'DATEPRD' in hist_data.columns:
                        hist_dates = hist_data['DATEPRD'].values
                        last_hist_date = hist_data['DATEPRD'].iloc[-1]
                    else:
                        # Fallback to date range if DATEPRD not available
                        last_date = st.session_state.df['DATEPRD'].max()
                        hist_dates = pd.date_range(end=last_date, periods=len(hist_data), freq='D')
                        last_hist_date = hist_dates[-1]
                    
                    fig.add_trace(go.Scatter(
                        x=hist_dates,
                        y=hist_data['BORE_OIL_VOL'],
                        mode='lines',
                        name='Historical Production',
                        line=dict(color='#2E86AB', width=2),
                        hovertemplate='<b>Historical</b><br>Date: %{x}<br>Production: %{y:.0f} bbl/day<extra></extra>'
                    ))
                    
                    # Forecast data
                    forecast_dates = pd.date_range(start=last_hist_date + pd.Timedelta(days=1), periods=len(st.session_state.forecast), freq='D')
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=st.session_state.forecast,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#A23B72', width=3, dash='dash'),
                        marker=dict(size=5, color='#A23B72'),
                        hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Production: %{y:.0f} bbl/day<extra></extra>'
                    ))
                    
                    # Add a connection line between historical and forecast
                    fig.add_trace(go.Scatter(
                        x=[last_hist_date, forecast_dates[0]],
                        y=[hist_data['BORE_OIL_VOL'].iloc[-1], st.session_state.forecast[0]],
                        mode='lines',
                        name='Transition',
                        line=dict(color='gray', width=1, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.update_layout(
                        title="Oil Production Forecast",
                        xaxis_title="Date",
                        yaxis_title="Production (bbl/day)",
                        height=400,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # History matching plot - Actual vs Predicted over time
                    fig = go.Figure()
                    
                    # Simply use numeric indices for the x-axis
                    test_indices = np.arange(len(predicted_test))
                    
                    # Add actual test data
                    fig.add_trace(go.Scatter(
                        x=test_indices,
                        y=actual_test,
                        mode='lines+markers',
                        name='Actual Production',
                        line=dict(color='#2E86AB', width=2),
                        marker=dict(size=4),
                        hovertemplate='<b>Actual</b><br>Index: %{x}<br>Production: %{y:.0f} bbl/day<extra></extra>'
                    ))
                    
                    # Add predicted test data
                    fig.add_trace(go.Scatter(
                        x=test_indices,
                        y=predicted_test,
                        mode='lines+markers',
                        name='Predicted Production',
                        line=dict(color='#A23B72', width=2, dash='dash'),
                        marker=dict(size=4),
                        hovertemplate='<b>Predicted</b><br>Index: %{x}<br>Production: %{y:.0f} bbl/day<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="History Matching: Actual vs Predicted Production",
                        xaxis_title="Time Index",
                        yaxis_title="Oil Production (bbl/day)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Actual vs Predicted scatter plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=actual_test,
                    y=predicted_test,
                    mode='markers',
                    name='Test Predictions',
                    marker=dict(color='#667eea', size=8, opacity=0.7),
                    hovertemplate='Actual: %{x:.0f}<br>Predicted: %{y:.0f}<extra></extra>'
                ))
                
                # Perfect prediction line
                min_val, max_val = min(actual_test), max(actual_test)
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='#f5576c', width=2, dash='dash'),
                    hoverinfo='skip'
                ))
                
                fig.update_layout(
                    title="Model Validation: Actual vs Predicted",
                    xaxis_title="Actual Production (bbl/day)",
                    yaxis_title="Predicted Production (bbl/day)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add comprehensive history matching plot
                st.subheader("📈 Complete History Matching")
                
                # Create comprehensive training + testing plot
                fig_complete = go.Figure()
                
                # Get training predictions for complete history matching
                if hasattr(st.session_state, 'forecaster') and st.session_state.forecaster.model is not None:
                    try:
                        # Make predictions on training data
                        X_train = st.session_state.get('X_train')
                        y_train = st.session_state.get('y_train')
                        lookback_days = st.session_state.get('lookback_days', 60)
                        
                        if X_train is not None and y_train is not None:
                            train_pred = st.session_state.forecaster.predict(X_train)
                            train_actual = st.session_state.forecaster.scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
                            
                            # Training data - actual
                            fig_complete.add_trace(go.Scatter(
                                x=np.arange(len(train_actual)),
                                y=train_actual,
                                mode='lines',
                                name='Training - Actual',
                                line=dict(color='#FFA500', width=2),  # Orange
                                hovertemplate='<b>Training Actual</b><br>Index: %{x}<br>Production: %{y:.0f} bbl/day<extra></extra>'
                            ))
                            
                            # Training data - predicted
                            fig_complete.add_trace(go.Scatter(
                                x=np.arange(len(train_pred.flatten())),
                                y=train_pred.flatten(),
                                mode='lines',
                                name='Training - Predicted',
                                line=dict(color='#2E86AB', width=4, dash='dot'),  # Increased width
                                hovertemplate='<b>Training Predicted</b><br>Index: %{x}<br>Production: %{y:.0f} bbl/day<extra></extra>'
                            ))
                            
                            # Testing data - actual
                            fig_complete.add_trace(go.Scatter(
                                x=np.arange(len(train_actual), len(train_actual) + len(actual_test)),
                                y=actual_test,
                                mode='lines',
                                name='Testing - Actual',
                                line=dict(color='#FFA500', width=2),  # Orange
                                hovertemplate='<b>Testing Actual</b><br>Index: %{x}<br>Production: %{y:.0f} bbl/day<extra></extra>'
                            ))
                            
                            # Testing data - predicted
                            fig_complete.add_trace(go.Scatter(
                                x=np.arange(len(train_actual), len(train_actual) + len(predicted_test)),
                                y=predicted_test,
                                mode='lines',
                                name='Testing - Predicted',
                                line=dict(color='#A23B72', width=4, dash='dot'),  # Increased width
                                hovertemplate='<b>Testing Predicted</b><br>Index: %{x}<br>Production: %{y:.0f} bbl/day<extra></extra>'
                            ))
                            
                            # Add vertical line to separate training and testing
                            split_index = len(train_actual) - 1
                            fig_complete.add_vline(
                                x=split_index,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text="Train/Test Split",
                                annotation_position="top"
                            )
                            
                            fig_complete.update_layout(
                                title="Complete History Matching: Training + Testing Data",
                                xaxis_title="Index",
                                yaxis_title="Oil Production (bbl/day)",
                                height=500,
                                hovermode='x unified',
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            st.plotly_chart(fig_complete, use_container_width=True)
                        else:
                            st.info("Training data not available for complete history matching plot.")
                    except Exception as e:
                        st.warning(f"Could not create complete history matching plot: {str(e)}")
                else:
                    st.info("Model not available for complete history matching.")
                
                # Model Performance Assessment
                st.header("📊 Model Performance Assessment")
                
                # Provide immediate model assessment
                performance_level = "Excellent" if metrics['RMSE'] < np.mean(st.session_state.clean_df['BORE_OIL_VOL']) * 0.1 else \
                                  "Good" if metrics['RMSE'] < np.mean(st.session_state.clean_df['BORE_OIL_VOL']) * 0.2 else \
                                  "Fair" if metrics['RMSE'] < np.mean(st.session_state.clean_df['BORE_OIL_VOL']) * 0.3 else "Poor"
                performance_color = {"Excellent": "🟢", "Good": "🟡", "Fair": "🟠", "Poor": "🔴"}[performance_level]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="padding: 1rem; background: rgba(46, 134, 171, 0.1); border-radius: 0.5rem; border-left: 4px solid #2E86AB;">
                        <h4>📊 Model Performance Summary</h4>
                        <p><strong>Performance Level:</strong> {performance_color} {performance_level}</p>
                        <p><strong>RMSE:</strong> {metrics['RMSE']:.1f} bbl/day ({metrics['RMSE']/np.mean(st.session_state.clean_df['BORE_OIL_VOL'])*100:.1f}% of avg production)</p>
                        <p><strong>MAE:</strong> {metrics['MAE']:.1f} bbl/day</p>
                        <p><strong>R²:</strong> {metrics.get('R2', 0):.3f} ({'Good fit' if metrics.get('R2', 0) > 0.7 else 'Moderate fit' if metrics.get('R2', 0) > 0.5 else 'Poor fit'})</p>
                        <p><strong>MAPE:</strong> {metrics.get('MAPE', 0):.1f}%</p>
                        <p><strong>Data Points Used:</strong> {metrics.get('Data_Points', 'N/A')} test samples</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Training history plot
                    if hasattr(st.session_state, 'history') and st.session_state.history is not None:
                        history = st.session_state.history
                        fig_loss = go.Figure()
                        
                        epochs_range = range(1, len(history.history['loss']) + 1)
                        
                        fig_loss.add_trace(go.Scatter(
                            x=list(epochs_range),
                            y=history.history['loss'],
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='#2E86AB', width=2)
                        ))
                        
                        if 'val_loss' in history.history:
                            fig_loss.add_trace(go.Scatter(
                                x=list(epochs_range),
                                y=history.history['val_loss'],
                                mode='lines',
                                name='Validation Loss',
                                line=dict(color='#A23B72', width=2)
                            ))
                        
                        fig_loss.update_layout(
                            title="Training Progress",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            height=300,
                            showlegend=True
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                
                # Recommendations section
                st.markdown(f"""
                <div style="padding: 1.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 0.8rem; margin: 1rem 0;">
                    <h4>💡 Model Optimization Recommendations</h4>
                    <div style="margin-top: 1rem;">
                """, unsafe_allow_html=True)
                
                # Generate recommendations based on performance
                recommendations = []
                
                if performance_level in ['Excellent', 'Good']:
                    recommendations.append("✅ Model is performing well and ready for production forecasting")
                    recommendations.append("🔄 Consider testing with different forecast horizons to validate consistency")
                else:
                    recommendations.append("⚠️ Model performance could be improved")
                    
                if metrics.get('R2', 0) < 0.5:
                    recommendations.append("📈 Low R² - Try increasing lookback window or using more relevant features")
                    recommendations.append("🔄 Consider switching to univariate mode for more stable predictions")
                
                if metrics.get('MAPE', 0) > 50:
                    recommendations.append("📊 High MAPE - Focus on RMSE and MAE metrics for evaluation")
                    recommendations.append("🎯 Consider filtering out zero production days for training")
                
                if performance_level == 'Poor':
                    recommendations.append("🔧 Try adjusting hyperparameters: increase epochs, change learning rate")
                    recommendations.append("📊 Check data quality and consider feature engineering")
                    recommendations.append("🎯 Experiment with different LSTM architectures")
                
                recommendations.append("💾 Save model parameters for future use and comparison")
                recommendations.append("📈 Monitor forecast accuracy over time and retrain as needed")
                
                for rec in recommendations:
                    st.markdown(f"<p style='margin: 0.5rem 0;'>{rec}</p>", unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                # Feature importance analysis
                if len(selected_features) > 1:
                    st.subheader("🔍 Feature Analysis")
                    
                    # Calculate basic correlations with target
                    correlations = df[selected_features].corr()['BORE_OIL_VOL'].drop('BORE_OIL_VOL')
                    
                    fig_corr = go.Figure(data=go.Bar(
                        x=correlations.index,
                        y=correlations.values,
                        marker_color=['#2E86AB' if x >= 0 else '#A23B72' for x in correlations.values]
                    ))
                    
                    fig_corr.update_layout(
                        title="Feature Correlation with Oil Production",
                        xaxis_title="Features",
                        yaxis_title="Correlation Coefficient",
                        height=400
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Download forecast data
                st.header("💾 Export Results")
                
                # Prepare forecast dataframe
                last_date = df['DATEPRD'].max()
                forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecasted_Production_bbl_day': st.session_state.forecast,
                    'Model_RMSE': metrics['RMSE'],
                    'Model_MAE': metrics['MAE'],
                    'Model_R2': metrics.get('R2', 0),
                    'Model_MAPE': metrics.get('MAPE', 0)
                })
                
                # Summary statistics
                summary_df = pd.DataFrame({
                    'Metric': ['RMSE', 'MAE', 'R²', 'MAPE', 'Avg_Forecast', 'Min_Forecast', 'Max_Forecast'],
                    'Value': [
                        metrics['RMSE'],
                        metrics['MAE'], 
                        metrics.get('R2', 0),
                        metrics.get('MAPE', 0),
                        np.mean(st.session_state.forecast),
                        np.min(st.session_state.forecast),
                        np.max(st.session_state.forecast)
                    ],
                    'Unit': ['bbl/day', 'bbl/day', '-', '%', 'bbl/day', 'bbl/day', 'bbl/day']
                })
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_forecast = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=csv_forecast,
                        file_name=f"oil_production_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Summary CSV",
                        data=csv_summary,
                        file_name=f"model_performance_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # Create a simple model configuration export
                    config_data = {
                        'timestamp': datetime.now().isoformat(),
                        'forecast_mode': st.session_state.get('forecast_mode', 'univariate'),
                        'lookback_days': st.session_state.get('lookback_days', 60),
                        'lstm_units': lstm_units,
                        'learning_rate': learning_rate,
                        'epochs': epochs,
                        'features': selected_features,
                        'performance': {
                            'rmse': metrics['RMSE'],
                            'mae': metrics['MAE'],
                            'r2': metrics.get('R2', 0),
                            'mape': metrics.get('MAPE', 0)
                        }
                    }
                    
                    import json
                    config_json = json.dumps(config_data, indent=2)
                    st.download_button(
                        label="⚙️ Download Config JSON",
                        data=config_json,
                        file_name=f"model_config_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )

        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👆 Please upload your production data CSV file to get started.")
        
        # Sample data format
        st.markdown("""
        ### 📋 Expected Data Format
        
        Your CSV should contain columns like:
        - **DATEPRD**: Production date (DD-MMM-YY format)
        - **BORE_OIL_VOL**: Oil production (bbl/day)
        - **AVG_DOWNHOLE_PRESSURE**: Downhole pressure
        - **AVG_WHP_P**: Wellhead pressure
        - **Total Water Injection**: Water injection volumes
        - **ON_STREAM_HRS**: Operating hours
        
        The tool will automatically detect and use available features for forecasting.
        """)

if __name__ == "__main__":
    main()