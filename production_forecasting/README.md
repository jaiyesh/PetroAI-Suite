# Production Forecasting with LSTM Neural Networks

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

## 📋 Project Overview

**PetroForecast AI** is an advanced oil production forecasting system using **LSTM (Long Short-Term Memory) neural networks**. This project provides both univariate and multivariate forecasting approaches with an interactive **Streamlit web interface** for real-time production predictions.

## 🎯 Objectives

- **Forecast** oil production using deep learning LSTM networks
- **Provide** both univariate and multivariate forecasting approaches
- **Deliver** interactive web-based forecasting interface
- **Optimize** model performance with advanced neural network techniques
- **Export** results for business intelligence and planning

## 🛠 Technical Implementation

### Deep Learning Architecture
- **LSTM Networks**: Advanced time series forecasting
- **Univariate Approach**: Production history-based predictions (more stable)
- **Multivariate Approach**: Including operational parameters
- **Real-time Training**: Early stopping and learning rate scheduling

### Neural Network Features
- **Sequential Architecture**: Layered LSTM design
- **Dropout Layers**: Preventing overfitting
- **Adam Optimizer**: Adaptive learning rate optimization
- **Custom Callbacks**: Early stopping and model checkpointing

## 📊 Key Features

### Interactive Web Interface
- **Streamlit Dashboard**: User-friendly web application
- **Real-time Forecasting**: Instant prediction generation
- **Interactive Visualizations**: Plotly-based charts and graphs
- **Model Configuration**: Adjustable parameters and settings
- **Export Capabilities**: CSV and JSON output formats

### Forecasting Capabilities
- **Flexible Time Windows**: 7-180 days lookback periods
- **Variable Horizons**: 1-180 days forecast periods
- **Multiple Metrics**: RMSE, MAE, R², MAPE evaluation
- **History Matching**: Model validation and performance assessment

## 🔧 Technical Specifications

### Data Parameters

#### Production Data
- **Daily Oil Production**: Primary time series target
- **Historical Trends**: Long-term production patterns
- **Data Quality**: Automated preprocessing and validation

#### Operational Parameters (Multivariate)
- **Downhole Pressure**: Reservoir pressure measurements
- **Wellhead Pressure**: Surface pressure readings
- **Water Injection**: Enhanced recovery operations
- **On-stream Hours**: Operational uptime metrics

### Model Configuration
- **LSTM Units**: 50-200 neurons per layer
- **Sequence Length**: Configurable lookback windows
- **Batch Size**: Optimized for memory and performance
- **Epochs**: Adaptive training with early stopping

## 🚀 Usage Instructions

### Running the Application
```bash
# Navigate to project directory
cd production_forecasting

# Install dependencies
pip install streamlit tensorflow pandas numpy plotly scikit-learn

# Launch the Streamlit app
streamlit run forecasting.py
```

### Accessing the Interface
- **Local URL**: http://localhost:8501
- **Web Interface**: Interactive dashboard with controls
- **Real-time Updates**: Instant model training and predictions

### Using the Application
1. **Upload Data**: CSV files with production history
2. **Configure Model**: Set parameters and training options
3. **Train Model**: Real-time LSTM training with progress indicators
4. **Generate Forecasts**: Predict future production rates
5. **Export Results**: Download predictions and configurations

## 📈 Key Features

### Model Performance
- **Multiple Metrics**: Comprehensive evaluation suite
- **Validation**: Train-test split and cross-validation
- **Uncertainty Quantification**: Prediction confidence intervals
- **Model Comparison**: Univariate vs. multivariate performance

### Interactive Elements
- **Parameter Sliders**: Real-time model configuration
- **Visual Feedback**: Live training progress and metrics
- **Data Exploration**: Interactive time series visualization
- **Export Options**: CSV, JSON, and chart downloads

## 🔍 Technical Details

### Data Processing
- **MinMax Scaling**: Feature normalization for neural networks
- **Sequence Generation**: Time window creation for LSTM input
- **Data Validation**: Quality checks and missing value handling
- **Feature Engineering**: Derived metrics and transformations

### Neural Network Architecture
```python
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(lookback, features)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])
```

### Training Process
- **Early Stopping**: Prevent overfitting with patience monitoring
- **Learning Rate Scheduling**: Adaptive rate adjustment
- **Model Checkpointing**: Save best performing models
- **Validation Monitoring**: Real-time performance tracking

## 🎯 Application Benefits

### Business Value
- **Production Planning**: Accurate forecasting for operations
- **Resource Allocation**: Optimize workforce and equipment
- **Financial Planning**: Revenue and cost projections
- **Risk Management**: Identify production decline trends

### Technical Advantages
- **Deep Learning**: State-of-the-art forecasting accuracy
- **Scalability**: Handle large datasets and multiple wells
- **Flexibility**: Configurable for different time horizons
- **User-Friendly**: No coding required for end users

## 📊 Visualization Features

### Interactive Charts
- **Time Series Plots**: Historical and predicted production
- **Performance Metrics**: Real-time training progress
- **Residual Analysis**: Model accuracy assessment
- **Uncertainty Bands**: Prediction confidence intervals

### Dashboard Components
- **Control Panel**: Parameter adjustment interface
- **Model Status**: Training progress and performance
- **Export Panel**: Download options and file formats
- **Help System**: Integrated documentation and tips

## 🔮 Future Enhancements

- **Real-time Data Integration**: Live production data feeds
- **Advanced Architectures**: Attention mechanisms and transformers
- **Ensemble Methods**: Multiple model combinations
- **Automated Hyperparameter Tuning**: Optimization algorithms
- **Mobile Interface**: Responsive design for field use

## 📚 Model Validation

### Performance Metrics
- **RMSE**: Root Mean Square Error for precision
- **MAE**: Mean Absolute Error for robustness
- **R²**: Coefficient of determination for fit quality
- **MAPE**: Mean Absolute Percentage Error for scale independence

### Validation Strategy
- **Train-Validation-Test Split**: 70-15-15 distribution
- **Time Series Cross-validation**: Temporal data splitting
- **Walk-forward Validation**: Realistic forecasting evaluation
- **Residual Analysis**: Error pattern identification

## 🛡️ Quality Assurance

### Data Quality
- **Missing Value Handling**: Interpolation and imputation
- **Outlier Detection**: Statistical anomaly identification
- **Data Consistency**: Temporal and logical validation
- **Format Standardization**: Consistent data structures

### Model Reliability
- **Convergence Monitoring**: Training stability assessment
- **Overfitting Prevention**: Regularization techniques
- **Performance Benchmarking**: Comparison with baseline models
- **Sensitivity Analysis**: Parameter robustness testing

---

**Note**: This application is designed for educational and research purposes. For production use, validate with domain experts and integrate with operational systems.

## 🤝 Contributing

Contributions welcome! Please submit issues, feature requests, or pull requests to enhance the forecasting capabilities.

---

*Part of the PetroAI-Suite - Comprehensive AI Solutions for Oil & Gas Industry*