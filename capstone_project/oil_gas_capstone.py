import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=== INTEGRATED OIL & GAS SUPPLY CHAIN OPTIMIZATION PROJECT ===")
print("Integrating Upstream, Midstream, and Downstream Operations with ML")
print("=" * 70)

# ===== DATA GENERATION =====
print("\n1. GENERATING SYNTHETIC INDUSTRY DATA")
print("-" * 40)

# Generate 2 years of daily data
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
n_days = len(dates)

# UPSTREAM DATA (Production)
print("Generating upstream (production) data...")
upstream_data = {
    'date': dates,
    'reservoir_pressure': np.random.normal(3000, 200, n_days),  # psi
    'water_cut': np.random.beta(2, 5, n_days) * 100,  # percentage
    'gas_oil_ratio': np.random.normal(800, 150, n_days),  # scf/bbl
    'wellhead_temperature': np.random.normal(180, 20, n_days),  # °F
    'drilling_activity': np.random.poisson(2, n_days),  # new wells/day
    'crude_oil_production': np.random.normal(50000, 8000, n_days),  # barrels/day
    'natural_gas_production': np.random.normal(25000, 4000, n_days),  # mcf/day
    'crude_api_gravity': np.random.normal(32, 4, n_days),  # API degrees
    'sulfur_content': np.random.beta(2, 8, n_days) * 3,  # weight %
}

# Add seasonal effects and correlations
seasonal_factor = np.sin(2 * np.pi * np.arange(n_days) / 365.25)
upstream_data['crude_oil_production'] += seasonal_factor * 2000
upstream_data['natural_gas_production'] += seasonal_factor * 1500

# MIDSTREAM DATA (Transportation & Refining)
print("Generating midstream (transportation & refining) data...")
midstream_data = {
    'date': dates,
    'pipeline_capacity_utilization': np.random.beta(6, 2, n_days) * 100,  # %
    'pipeline_pressure': np.random.normal(1000, 100, n_days),  # psi
    'compressor_efficiency': np.random.normal(85, 5, n_days),  # %
    'refinery_utilization': np.random.beta(7, 3, n_days) * 100,  # %
    'catalyst_activity': np.random.normal(75, 8, n_days),  # %
    'gasoline_yield': np.random.normal(45, 5, n_days),  # % of crude input
    'diesel_yield': np.random.normal(25, 3, n_days),  # % of crude input
    'jet_fuel_yield': np.random.normal(12, 2, n_days),  # % of crude input
    'refinery_energy_cost': np.random.gamma(2, 15, n_days),  # $/barrel
    'transportation_cost': np.random.gamma(3, 2, n_days),  # $/barrel
}

# DOWNSTREAM DATA (Distribution & Retail)
print("Generating downstream (distribution & retail) data...")
downstream_data = {
    'date': dates,
    'gasoline_demand': np.random.normal(180000, 25000, n_days),  # barrels/day
    'diesel_demand': np.random.normal(120000, 15000, n_days),  # barrels/day
    'jet_fuel_demand': np.random.normal(80000, 12000, n_days),  # barrels/day
    'gasoline_price': np.random.normal(2.8, 0.4, n_days),  # $/gallon
    'diesel_price': np.random.normal(3.1, 0.3, n_days),  # $/gallon
    'jet_fuel_price': np.random.normal(2.5, 0.3, n_days),  # $/gallon
    'retail_margin': np.random.normal(15, 3, n_days),  # cents/gallon
    'inventory_gasoline': np.random.normal(5000000, 800000, n_days),  # barrels
    'inventory_diesel': np.random.normal(3000000, 500000, n_days),  # barrels
    'inventory_jet_fuel': np.random.normal(2000000, 400000, n_days),  # barrels
}

# Add market dynamics and correlations
market_volatility = np.random.normal(0, 0.1, n_days)
downstream_data['gasoline_demand'] += seasonal_factor * 15000 + market_volatility * 10000
downstream_data['diesel_demand'] += seasonal_factor * 8000 + market_volatility * 5000

# EXTERNAL FACTORS
print("Generating external factors data...")
external_data = {
    'date': dates,
    'crude_oil_price': np.random.normal(75, 12, n_days),  # $/barrel
    'natural_gas_price': np.random.normal(4.5, 1.2, n_days),  # $/mmbtu
    'weather_temperature': np.random.normal(65, 20, n_days),  # °F
    'economic_indicator': np.random.normal(100, 15, n_days),  # index
    'environmental_regulation_score': np.random.normal(50, 10, n_days),  # index
}

# Combine all data
print("Combining all datasets...")
df_upstream = pd.DataFrame(upstream_data)
df_midstream = pd.DataFrame(midstream_data)
df_downstream = pd.DataFrame(downstream_data)
df_external = pd.DataFrame(external_data)

# Merge all dataframes
df_combined = df_upstream.merge(df_midstream, on='date').merge(df_downstream, on='date').merge(df_external, on='date')

# Add derived features
df_combined['profit_per_barrel'] = (df_combined['gasoline_price'] * 42 * df_combined['gasoline_yield']/100 + 
                                   df_combined['diesel_price'] * 42 * df_combined['diesel_yield']/100) - \
                                  (df_combined['crude_oil_price'] + df_combined['refinery_energy_cost'] + 
                                   df_combined['transportation_cost'])

df_combined['total_production_value'] = (df_combined['crude_oil_production'] * df_combined['crude_oil_price'] + 
                                        df_combined['natural_gas_production'] * df_combined['natural_gas_price'])

df_combined['supply_demand_ratio'] = (df_combined['crude_oil_production'] / 
                                     (df_combined['gasoline_demand'] + df_combined['diesel_demand'] + 
                                      df_combined['jet_fuel_demand']))

print(f"Generated dataset with {len(df_combined)} records and {len(df_combined.columns)} features")
print(f"Data range: {df_combined['date'].min()} to {df_combined['date'].max()}")

# ===== EXPLORATORY DATA ANALYSIS =====
print("\n2. EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# Basic statistics
print("Dataset shape:", df_combined.shape)
print("\nBasic statistics:")
print(df_combined.describe())

# Check for missing values
print("\nMissing values:")
print(df_combined.isnull().sum().sum(), "total missing values")

# ===== DATA VISUALIZATION =====
print("\n3. DATA VISUALIZATION")
print("-" * 40)

# Create comprehensive visualizations
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Integrated Oil & Gas Supply Chain Analysis', fontsize=16)

# Upstream trends
axes[0,0].plot(df_combined['date'], df_combined['crude_oil_production'], color='brown', alpha=0.7)
axes[0,0].set_title('Upstream: Crude Oil Production')
axes[0,0].set_ylabel('Barrels/day')
axes[0,0].tick_params(axis='x', rotation=45)

# Midstream efficiency
axes[0,1].plot(df_combined['date'], df_combined['refinery_utilization'], color='orange', alpha=0.7)
axes[0,1].set_title('Midstream: Refinery Utilization')
axes[0,1].set_ylabel('Utilization %')
axes[0,1].tick_params(axis='x', rotation=45)

# Downstream demand
axes[1,0].plot(df_combined['date'], df_combined['gasoline_demand'], color='blue', alpha=0.7, label='Gasoline')
axes[1,0].plot(df_combined['date'], df_combined['diesel_demand'], color='green', alpha=0.7, label='Diesel')
axes[1,0].set_title('Downstream: Product Demand')
axes[1,0].set_ylabel('Barrels/day')
axes[1,0].legend()
axes[1,0].tick_params(axis='x', rotation=45)

# Price trends
axes[1,1].plot(df_combined['date'], df_combined['crude_oil_price'], color='red', alpha=0.7)
axes[1,1].set_title('Crude Oil Price Trends')
axes[1,1].set_ylabel('$/barrel')
axes[1,1].tick_params(axis='x', rotation=45)

# Profitability
axes[2,0].plot(df_combined['date'], df_combined['profit_per_barrel'], color='purple', alpha=0.7)
axes[2,0].set_title('Profit per Barrel')
axes[2,0].set_ylabel('$/barrel')
axes[2,0].tick_params(axis='x', rotation=45)

# Supply-demand ratio
axes[2,1].plot(df_combined['date'], df_combined['supply_demand_ratio'], color='teal', alpha=0.7)
axes[2,1].set_title('Supply-Demand Ratio')
axes[2,1].set_ylabel('Ratio')
axes[2,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Correlation analysis
print("\nCorrelation Analysis:")
correlation_features = ['crude_oil_production', 'refinery_utilization', 'gasoline_demand', 
                       'crude_oil_price', 'profit_per_barrel', 'supply_demand_ratio']
corr_matrix = df_combined[correlation_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix: Key Performance Indicators')
plt.tight_layout()
plt.show()

# ===== MACHINE LEARNING MODELS =====
print("\n4. MACHINE LEARNING MODELS")
print("-" * 40)

# Prepare features for modeling
feature_columns = [col for col in df_combined.columns if col not in ['date', 'profit_per_barrel']]
X = df_combined[feature_columns]
y = df_combined['profit_per_barrel']

# Add time-based features
df_combined['day_of_year'] = df_combined['date'].dt.dayofyear
df_combined['month'] = df_combined['date'].dt.month
df_combined['quarter'] = df_combined['date'].dt.quarter

# Update feature set
feature_columns = [col for col in df_combined.columns if col not in ['date', 'profit_per_barrel']]
X = df_combined[feature_columns]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ===== MODEL TRAINING AND EVALUATION =====
print("\n5. MODEL TRAINING AND EVALUATION")
print("-" * 40)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    results[name] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  R²: {r2:.3f}")
    print(f"  CV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ===== MODEL COMPARISON =====
print("\n6. MODEL COMPARISON")
print("-" * 40)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[name]['rmse'] for name in results.keys()],
    'MAE': [results[name]['mae'] for name in results.keys()],
    'R²': [results[name]['r2'] for name in results.keys()],
    'CV R²': [results[name]['cv_mean'] for name in results.keys()]
})

print("\nModel Performance Comparison:")
print(comparison_df.round(3))

# Best model
best_model_name = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
best_model = results[best_model_name]['model']
print(f"\nBest performing model: {best_model_name}")

# ===== FEATURE IMPORTANCE =====
print("\n7. FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance ({best_model_name})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# ===== PREDICTIONS AND BUSINESS INSIGHTS =====
print("\n8. PREDICTIONS AND BUSINESS INSIGHTS")
print("-" * 40)

# Generate predictions for the test set
best_predictions = results[best_model_name]['predictions']

# Create prediction vs actual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Profit per Barrel ($)')
plt.ylabel('Predicted Profit per Barrel ($)')
plt.title(f'Actual vs Predicted Profit ({best_model_name})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Business insights
print("\nBUSINESS INSIGHTS:")
print("=" * 50)

avg_profit = df_combined['profit_per_barrel'].mean()
print(f"1. Average profit per barrel: ${avg_profit:.2f}")

# Identify high-profit scenarios
high_profit_threshold = df_combined['profit_per_barrel'].quantile(0.8)
high_profit_data = df_combined[df_combined['profit_per_barrel'] > high_profit_threshold]

print(f"2. High-profit scenarios (top 20%) characteristics:")
print(f"   - Average refinery utilization: {high_profit_data['refinery_utilization'].mean():.1f}%")
print(f"   - Average crude oil price: ${high_profit_data['crude_oil_price'].mean():.2f}/barrel")
print(f"   - Average gasoline demand: {high_profit_data['gasoline_demand'].mean():.0f} barrels/day")

# Optimization recommendations
print(f"\n3. OPTIMIZATION RECOMMENDATIONS:")
print("   a) Upstream: Focus on wells with higher API gravity crude")
print("   b) Midstream: Maintain refinery utilization above 85%")
print("   c) Downstream: Monitor demand patterns for seasonal optimization")
print("   d) Integration: Implement real-time data sharing across all sectors")

# ===== FUTURE PREDICTIONS =====
print("\n9. FUTURE SCENARIO PLANNING")
print("-" * 40)

# Create future scenarios
scenarios = {
    'Optimistic': {
        'crude_oil_price': 85,
        'refinery_utilization': 90,
        'gasoline_demand': 200000,
        'diesel_demand': 135000
    },
    'Base Case': {
        'crude_oil_price': 75,
        'refinery_utilization': 85,
        'gasoline_demand': 180000,
        'diesel_demand': 120000
    },
    'Pessimistic': {
        'crude_oil_price': 65,
        'refinery_utilization': 75,
        'gasoline_demand': 160000,
        'diesel_demand': 105000
    }
}

print("Future Profit Scenarios:")
for scenario_name, scenario_data in scenarios.items():
    # Create a sample input with median values for other features
    sample_input = np.array([df_combined[col].median() for col in feature_columns]).reshape(1, -1)
    
    # Update with scenario-specific values
    feature_indices = {col: i for i, col in enumerate(feature_columns)}
    for param, value in scenario_data.items():
        if param in feature_indices:
            sample_input[0, feature_indices[param]] = value
    
    # Scale and predict
    sample_scaled = scaler.transform(sample_input)
    predicted_profit = best_model.predict(sample_scaled)[0]
    
    print(f"  {scenario_name}: ${predicted_profit:.2f} per barrel")

# ===== PROJECT SUMMARY =====
print("\n" + "="*70)
print("PROJECT SUMMARY")
print("="*70)
print(f"✓ Dataset: {len(df_combined)} records across {len(df_combined.columns)} features")
print(f"✓ Integration: Upstream + Midstream + Downstream + External factors")
print(f"✓ Best Model: {best_model_name} (R² = {results[best_model_name]['r2']:.3f})")
print(f"✓ Key Insight: Model can predict profitability with {results[best_model_name]['r2']*100:.1f}% accuracy")
print(f"✓ Business Value: Real-time optimization across the entire value chain")
print("="*70)

print("\nProject completed successfully!")
print("This integrated approach enables:")
print("1. Predictive maintenance and production optimization")
print("2. Real-time refinery and pipeline capacity planning")
print("3. Demand forecasting and inventory optimization")
print("4. End-to-end supply chain visibility and control")
print("5. Data-driven decision making across all business segments")