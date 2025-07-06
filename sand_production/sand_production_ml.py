"""
Sand Production Prediction Using Machine Learning
=================================================

A comprehensive petroleum engineering project for predicting sand production in oil wells
using various machine learning algorithms and realistic geological/engineering parameters.

Author: Petroleum Engineering Expert
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class SandProductionDataGenerator:
    """
    Generates realistic sand production data based on petroleum engineering principles
    """
    
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        
    def generate_data(self):
        """
        Generate synthetic but realistic petroleum engineering data
        """
        print("Generating realistic petroleum engineering data...")
        
        # Reservoir Properties
        permeability = np.random.lognormal(mean=2.0, sigma=1.5, size=self.n_samples)  # mD
        porosity = np.random.normal(0.15, 0.05, self.n_samples)  # fraction
        porosity = np.clip(porosity, 0.05, 0.35)
        
        # Rock Mechanical Properties
        compressive_strength = np.random.normal(25, 8, self.n_samples)  # MPa
        compressive_strength = np.clip(compressive_strength, 5, 50)
        
        cohesion = np.random.normal(2.5, 0.8, self.n_samples)  # MPa
        cohesion = np.clip(cohesion, 0.5, 6.0)
        
        friction_angle = np.random.normal(30, 5, self.n_samples)  # degrees
        friction_angle = np.clip(friction_angle, 20, 45)
        
        # Formation Properties
        grain_size = np.random.lognormal(mean=-1.5, sigma=0.8, size=self.n_samples)  # mm
        grain_size = np.clip(grain_size, 0.05, 2.0)
        
        cement_quality = np.random.uniform(0.3, 0.9, self.n_samples)  # quality factor (0-1)
        
        clay_content = np.random.beta(2, 5, self.n_samples) * 0.3  # fraction
        
        # Reservoir Conditions
        reservoir_pressure = np.random.normal(250, 50, self.n_samples)  # bar
        reservoir_pressure = np.clip(reservoir_pressure, 100, 400)
        
        temperature = np.random.normal(80, 15, self.n_samples)  # Celsius
        temperature = np.clip(temperature, 40, 120)
        
        water_saturation = np.random.beta(2, 3, self.n_samples)  # fraction
        
        # Well Completion Factors
        perforation_density = np.random.normal(12, 3, self.n_samples)  # shots per meter
        perforation_density = np.clip(perforation_density, 4, 20)
        
        completion_types = np.random.choice(['OpenHole', 'CasedHole', 'Gravel_Pack'], 
                                          size=self.n_samples, p=[0.3, 0.4, 0.3])
        
        # Production Parameters
        flow_rate = np.random.lognormal(mean=3.5, sigma=0.8, size=self.n_samples)  # m3/day
        flow_rate = np.clip(flow_rate, 10, 200)
        
        drawdown_pressure = np.random.normal(15, 5, self.n_samples)  # bar
        drawdown_pressure = np.clip(drawdown_pressure, 2, 35)
        
        # Fluid Properties
        oil_viscosity = np.random.lognormal(mean=1.0, sigma=0.6, size=self.n_samples)  # cp
        oil_viscosity = np.clip(oil_viscosity, 0.5, 10)
        
        oil_density = np.random.normal(850, 50, self.n_samples)  # kg/m3
        oil_density = np.clip(oil_density, 750, 950)
        
        # Calculate Sand Production Rate using petroleum engineering correlations
        sand_production_rate = self._calculate_sand_production(
            permeability, porosity, compressive_strength, cohesion, friction_angle,
            grain_size, cement_quality, clay_content, reservoir_pressure, 
            temperature, water_saturation, perforation_density, completion_types,
            flow_rate, drawdown_pressure, oil_viscosity, oil_density
        )
        
        # Create DataFrame
        data = pd.DataFrame({
            'Permeability_mD': permeability,
            'Porosity': porosity,
            'Compressive_Strength_MPa': compressive_strength,
            'Cohesion_MPa': cohesion,
            'Friction_Angle_deg': friction_angle,
            'Grain_Size_mm': grain_size,
            'Cement_Quality': cement_quality,
            'Clay_Content': clay_content,
            'Reservoir_Pressure_bar': reservoir_pressure,
            'Temperature_C': temperature,
            'Water_Saturation': water_saturation,
            'Perforation_Density_spm': perforation_density,
            'Completion_Type': completion_types,
            'Flow_Rate_m3day': flow_rate,
            'Drawdown_Pressure_bar': drawdown_pressure,
            'Oil_Viscosity_cp': oil_viscosity,
            'Oil_Density_kgm3': oil_density,
            'Sand_Production_Rate_kgday': sand_production_rate
        })
        
        print(f"Generated {len(data)} samples with {len(data.columns)} features")
        return data
    
    def _calculate_sand_production(self, perm, por, comp_str, cohesion, friction_angle,
                                 grain_size, cement_qual, clay_content, res_press, temp,
                                 water_sat, perf_density, completion_type, flow_rate,
                                 drawdown, viscosity, density):
        """
        Calculate sand production using petroleum engineering correlations
        """
        # Normalize completion type effect
        completion_factor = np.where(completion_type == 'OpenHole', 1.2,
                           np.where(completion_type == 'CasedHole', 1.0, 0.6))
        
        # Critical velocity factor (Veeken et al.)
        critical_velocity = (comp_str * cohesion) / (grain_size * density)
        
        # Actual velocity
        actual_velocity = flow_rate / (perm * drawdown)
        
        # Velocity ratio
        velocity_ratio = actual_velocity / critical_velocity
        
        # Rock strength factor
        rock_strength_factor = comp_str * cohesion * np.cos(np.radians(friction_angle))
        
        # Formation quality factor
        formation_factor = cement_qual * (1 - clay_content) * (1 - water_sat)
        
        # Sand production calculation
        base_sand_rate = (
            velocity_ratio * 
            (grain_size / comp_str) * 
            (flow_rate / rock_strength_factor) * 
            completion_factor * 
            (1 / formation_factor) *
            (viscosity / 10) *  # Viscosity effect
            (drawdown / res_press) *  # Pressure depletion effect
            100  # Scale factor
        )
        
        # Add some realistic noise and ensure non-negative values
        noise = np.random.normal(1, 0.3, len(base_sand_rate))
        sand_rate = base_sand_rate * noise
        sand_rate = np.clip(sand_rate, 0, 500)  # Realistic upper limit
        
        return sand_rate

class SandProductionPredictor:
    """
    Machine Learning model for sand production prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.feature_importance = None
        
    def preprocess_data(self, data):
        """
        Preprocess the data for machine learning
        """
        print("Preprocessing data...")
        
        # Separate features and target
        X = data.drop('Sand_Production_Rate_kgday', axis=1)
        y = data['Sand_Production_Rate_kgday']
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_processed = X.copy()
        
        for col in categorical_cols:
            X_processed[col] = self.label_encoder.fit_transform(X[col])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns)
        
        return X_scaled, y
    
    def train_models(self, X, y):
        """
        Train multiple machine learning models
        """
        print("Training machine learning models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=100, gamma=0.1),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'cv_rmse': cv_rmse,
                'y_pred_test': y_pred_test
            }
        
        self.models = results
        self.X_test = X_test
        self.y_test = y_test
        
        # Select best model based on test R²
        best_model_name = max(results, key=lambda x: results[x]['test_r2'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"Best model: {best_model_name} (R² = {results[best_model_name]['test_r2']:.4f})")
        
        return results
    
    def get_feature_importance(self, X):
        """
        Get feature importance from the best model
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_names = X.columns
            
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return self.feature_importance
        else:
            print("Best model doesn't support feature importance")
            return None
    
    def plot_results(self):
        """
        Create comprehensive visualizations
        """
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Performance Comparison
        plt.subplot(2, 3, 1)
        model_names = list(self.models.keys())
        test_r2_scores = [self.models[name]['test_r2'] for name in model_names]
        test_rmse_scores = [self.models[name]['test_rmse'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        plt.bar(x_pos, test_r2_scores, alpha=0.7, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison (R²)')
        plt.xticks(x_pos, model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 2. RMSE Comparison
        plt.subplot(2, 3, 2)
        plt.bar(x_pos, test_rmse_scores, alpha=0.7, color='lightcoral')
        plt.xlabel('Models')
        plt.ylabel('RMSE (kg/day)')
        plt.title('Model Performance Comparison (RMSE)')
        plt.xticks(x_pos, model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 3. Actual vs Predicted for Best Model
        plt.subplot(2, 3, 3)
        best_predictions = self.models[self.best_model_name]['y_pred_test']
        plt.scatter(self.y_test, best_predictions, alpha=0.6, color='green')
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Sand Production (kg/day)')
        plt.ylabel('Predicted Sand Production (kg/day)')
        plt.title(f'Actual vs Predicted ({self.best_model_name})')
        plt.grid(True, alpha=0.3)
        
        # 4. Feature Importance
        if self.feature_importance is not None:
            plt.subplot(2, 3, 4)
            top_features = self.feature_importance.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.grid(True, alpha=0.3)
        
        # 5. Residuals Plot
        plt.subplot(2, 3, 5)
        residuals = self.y_test - best_predictions
        plt.scatter(best_predictions, residuals, alpha=0.6, color='purple')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Sand Production (kg/day)')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # 6. Distribution of Sand Production
        plt.subplot(2, 3, 6)
        plt.hist(self.y_test, bins=30, alpha=0.7, color='orange', label='Actual')
        plt.hist(best_predictions, bins=30, alpha=0.7, color='blue', label='Predicted')
        plt.xlabel('Sand Production Rate (kg/day)')
        plt.ylabel('Frequency')
        plt.title('Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_model_summary(self):
        """
        Print comprehensive model performance summary
        """
        print("\n" + "="*80)
        print("SAND PRODUCTION PREDICTION - MODEL PERFORMANCE SUMMARY")
        print("="*80)
        
        print(f"\n{'Model':<25} {'Train R²':<12} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12} {'CV RMSE':<12}")
        print("-" * 85)
        
        for name, results in self.models.items():
            print(f"{name:<25} {results['train_r2']:<12.4f} {results['test_r2']:<12.4f} "
                  f"{results['test_rmse']:<12.2f} {results['test_mae']:<12.2f} {results['cv_rmse']:<12.2f}")
        
        print(f"\nBest Performing Model: {self.best_model_name}")
        best_results = self.models[self.best_model_name]
        print(f"Test R² Score: {best_results['test_r2']:.4f}")
        print(f"Test RMSE: {best_results['test_rmse']:.2f} kg/day")
        print(f"Test MAE: {best_results['test_mae']:.2f} kg/day")
        
        if self.feature_importance is not None:
            print(f"\nTop 5 Most Important Features:")
            for i, row in self.feature_importance.head().iterrows():
                print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")

def main():
    """
    Main function to run the complete sand production prediction project
    """
    print("🛢️  SAND PRODUCTION PREDICTION USING MACHINE LEARNING")
    print("=" * 60)
    print("A Petroleum Engineering AI Project")
    print("=" * 60)
    
    # Step 1: Generate Data
    print("\n📊 STEP 1: DATA GENERATION")
    data_generator = SandProductionDataGenerator(n_samples=1000)
    data = data_generator.generate_data()
    
    # Display basic statistics
    print(f"\nDataset Shape: {data.shape}")
    print(f"Target Variable: Sand Production Rate (kg/day)")
    print(f"Mean Sand Production: {data['Sand_Production_Rate_kgday'].mean():.2f} kg/day")
    print(f"Std Sand Production: {data['Sand_Production_Rate_kgday'].std():.2f} kg/day")
    print(f"Range: {data['Sand_Production_Rate_kgday'].min():.2f} - {data['Sand_Production_Rate_kgday'].max():.2f} kg/day")
    
    # Step 2: Data Preprocessing and Model Training
    print("\n🤖 STEP 2: MACHINE LEARNING MODEL TRAINING")
    predictor = SandProductionPredictor()
    X, y = predictor.preprocess_data(data)
    results = predictor.train_models(X, y)
    
    # Step 3: Feature Importance Analysis
    print("\n📈 STEP 3: FEATURE IMPORTANCE ANALYSIS")
    feature_importance = predictor.get_feature_importance(X)
    
    # Step 4: Results and Visualization
    print("\n📊 STEP 4: RESULTS AND VISUALIZATION")
    predictor.print_model_summary()
    
    # Step 5: Plot Results
    print("\n📈 STEP 5: GENERATING PLOTS")
    predictor.plot_results()
    
    # Save results
    print("\n💾 STEP 6: SAVING RESULTS")
    data.to_csv('sand_production_data.csv', index=False)
    print("Data saved to 'sand_production_data.csv'")
    
    if feature_importance is not None:
        feature_importance.to_csv('feature_importance.csv', index=False)
        print("Feature importance saved to 'feature_importance.csv'")
    
    print("\n✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return data, predictor, results

# Run the project
if __name__ == "__main__":
    data, predictor, results = main()