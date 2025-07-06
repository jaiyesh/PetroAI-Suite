import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class HydraulicFracturingOptimization:
    """
    Comprehensive ML-based hydraulic fracturing optimization system
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.optimization_model = None
        self.classification_model = None
        self.refrac_model = None
        self.data = None
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic petroleum engineering data for fracturing analysis
        """
        print("Generating synthetic petroleum engineering data...")
        
        # Geological parameters
        porosity = np.random.normal(0.12, 0.04, n_samples)
        porosity = np.clip(porosity, 0.05, 0.25)
        
        permeability = np.random.lognormal(-2, 1.5, n_samples)  # mD
        permeability = np.clip(permeability, 0.001, 50)
        
        young_modulus = np.random.normal(4.5e6, 1e6, n_samples)  # psi
        young_modulus = np.clip(young_modulus, 2e6, 8e6)
        
        poisson_ratio = np.random.normal(0.25, 0.05, n_samples)
        poisson_ratio = np.clip(poisson_ratio, 0.15, 0.35)
        
        # Stress parameters
        min_horizontal_stress = np.random.normal(6000, 1000, n_samples)  # psi
        max_horizontal_stress = min_horizontal_stress + np.random.normal(2000, 500, n_samples)
        vertical_stress = np.random.normal(1.1, 0.1, n_samples) * (min_horizontal_stress + max_horizontal_stress) / 2
        
        # Formation parameters
        net_thickness = np.random.normal(150, 30, n_samples)  # ft
        net_thickness = np.clip(net_thickness, 50, 300)
        
        toc = np.random.normal(8, 2, n_samples)  # Total Organic Content %
        toc = np.clip(toc, 2, 15)
        
        # Fracturing parameters
        proppant_concentration = np.random.normal(2, 0.5, n_samples)  # lb/gal
        proppant_concentration = np.clip(proppant_concentration, 0.5, 4)
        
        fluid_rate = np.random.normal(80, 20, n_samples)  # bbl/min
        fluid_rate = np.clip(fluid_rate, 30, 150)
        
        fluid_volume = np.random.normal(5000, 1500, n_samples)  # bbl
        fluid_volume = np.clip(fluid_volume, 2000, 10000)
        
        stage_length = np.random.normal(200, 50, n_samples)  # ft
        stage_length = np.clip(stage_length, 100, 400)
        
        # Calculate derived parameters
        brittleness_index = (young_modulus / 1e6 + (1 - poisson_ratio) * 20) / 2
        stress_ratio = min_horizontal_stress / max_horizontal_stress
        
        # Generate target variables
        
        # 1. Fracture half-length (optimization target)
        fracture_half_length = (
            50 + 
            0.3 * young_modulus / 1e6 +
            0.2 * fluid_volume / 1000 +
            0.15 * proppant_concentration * 50 +
            0.1 * brittleness_index * 30 +
            np.random.normal(0, 20, n_samples)
        )
        fracture_half_length = np.clip(fracture_half_length, 50, 500)
        
        # 2. Fracture intensity classification (Low, Medium, High)
        intensity_score = (
            0.3 * brittleness_index +
            0.2 * toc / 10 +
            0.2 * (1 - stress_ratio) +
            0.15 * permeability / 10 +
            0.15 * porosity * 10 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        fracture_intensity = np.where(intensity_score < 0.4, 'Low',
                                    np.where(intensity_score < 0.7, 'Medium', 'High'))
        
        # 3. Re-frac feasibility (binary classification)
        # Consider production decline, reservoir pressure, completion quality
        production_decline = np.random.exponential(0.3, n_samples)
        reservoir_pressure_ratio = np.random.normal(0.6, 0.2, n_samples)
        reservoir_pressure_ratio = np.clip(reservoir_pressure_ratio, 0.2, 0.9)
        
        refrac_probability = (
            0.4 * production_decline +
            0.3 * (1 - reservoir_pressure_ratio) +
            0.2 * (fracture_half_length / 500) +
            0.1 * (permeability / 10) +
            np.random.normal(0, 0.1, n_samples)
        )
        
        refrac_feasible = (refrac_probability > 0.5).astype(int)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'porosity': porosity,
            'permeability': permeability,
            'young_modulus': young_modulus,
            'poisson_ratio': poisson_ratio,
            'min_horizontal_stress': min_horizontal_stress,
            'max_horizontal_stress': max_horizontal_stress,
            'vertical_stress': vertical_stress,
            'net_thickness': net_thickness,
            'toc': toc,
            'proppant_concentration': proppant_concentration,
            'fluid_rate': fluid_rate,
            'fluid_volume': fluid_volume,
            'stage_length': stage_length,
            'brittleness_index': brittleness_index,
            'stress_ratio': stress_ratio,
            'production_decline': production_decline,
            'reservoir_pressure_ratio': reservoir_pressure_ratio,
            'fracture_half_length': fracture_half_length,
            'fracture_intensity': fracture_intensity,
            'refrac_feasible': refrac_feasible
        })
        
        print(f"Generated {n_samples} synthetic data points")
        return self.data
    
    def visualize_data(self):
        """
        Create comprehensive data visualization
        """
        print("Creating data visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Geological parameters
        axes[0, 0].scatter(self.data['porosity'], self.data['permeability'], 
                          c=self.data['fracture_half_length'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel('Porosity')
        axes[0, 0].set_ylabel('Permeability (mD)')
        axes[0, 0].set_title('Porosity vs Permeability')
        axes[0, 0].set_yscale('log')
        
        # Stress analysis
        axes[0, 1].scatter(self.data['min_horizontal_stress'], self.data['max_horizontal_stress'], 
                          c=self.data['fracture_half_length'], cmap='viridis', alpha=0.6)
        axes[0, 1].set_xlabel('Min Horizontal Stress (psi)')
        axes[0, 1].set_ylabel('Max Horizontal Stress (psi)')
        axes[0, 1].set_title('Stress State Analysis')
        
        # Fracture intensity distribution
        intensity_counts = self.data['fracture_intensity'].value_counts()
        axes[0, 2].pie(intensity_counts.values, labels=intensity_counts.index, autopct='%1.1f%%')
        axes[0, 2].set_title('Fracture Intensity Distribution')
        
        # Young's modulus vs fracture length
        axes[1, 0].scatter(self.data['young_modulus']/1e6, self.data['fracture_half_length'], 
                          c=self.data['brittleness_index'], cmap='plasma', alpha=0.6)
        axes[1, 0].set_xlabel('Young\'s Modulus (MMpsi)')
        axes[1, 0].set_ylabel('Fracture Half-Length (ft)')
        axes[1, 0].set_title('Young\'s Modulus vs Fracture Length')
        
        # Fluid volume vs fracture length
        axes[1, 1].scatter(self.data['fluid_volume'], self.data['fracture_half_length'], 
                          c=self.data['proppant_concentration'], cmap='coolwarm', alpha=0.6)
        axes[1, 1].set_xlabel('Fluid Volume (bbl)')
        axes[1, 1].set_ylabel('Fracture Half-Length (ft)')
        axes[1, 1].set_title('Fluid Volume vs Fracture Length')
        
        # Re-frac feasibility
        refrac_counts = self.data['refrac_feasible'].value_counts()
        axes[1, 2].pie(refrac_counts.values, labels=['Not Feasible', 'Feasible'], autopct='%1.1f%%')
        axes[1, 2].set_title('Re-Frac Feasibility Distribution')
        
        # Brittleness index distribution
        axes[2, 0].hist(self.data['brittleness_index'], bins=30, alpha=0.7, color='skyblue')
        axes[2, 0].set_xlabel('Brittleness Index')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('Brittleness Index Distribution')
        
        # TOC vs Production decline
        axes[2, 1].scatter(self.data['toc'], self.data['production_decline'], 
                          c=self.data['refrac_feasible'], cmap='RdYlBu', alpha=0.6)
        axes[2, 1].set_xlabel('TOC (%)')
        axes[2, 1].set_ylabel('Production Decline')
        axes[2, 1].set_title('TOC vs Production Decline')
        
        # Correlation heatmap
        correlation_features = ['porosity', 'permeability', 'young_modulus', 'brittleness_index', 
                               'fluid_volume', 'proppant_concentration', 'fracture_half_length']
        corr_matrix = self.data[correlation_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[2, 2])
        axes[2, 2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
    
    def train_optimization_model(self):
        """
        Train ML model for fracture half-length optimization
        """
        print("Training fracture optimization model...")
        
        # Features for optimization
        opt_features = [
            'porosity', 'permeability', 'young_modulus', 'poisson_ratio',
            'min_horizontal_stress', 'max_horizontal_stress', 'net_thickness',
            'toc', 'proppant_concentration', 'fluid_rate', 'fluid_volume',
            'stage_length', 'brittleness_index', 'stress_ratio'
        ]
        
        X = self.data[opt_features]
        y = self.data['fracture_half_length']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.optimization_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        self.optimization_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.optimization_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Optimization Model Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"R²: {r2:.3f}")
        print(f"RMSE: {np.sqrt(mse):.2f} ft")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': opt_features,
            'importance': self.optimization_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        return self.optimization_model
    
    def train_classification_model(self):
        """
        Train ML model for fracture intensity classification
        """
        print("\nTraining fracture intensity classification model...")
        
        # Features for classification
        class_features = [
            'porosity', 'permeability', 'young_modulus', 'poisson_ratio',
            'brittleness_index', 'toc', 'stress_ratio', 'net_thickness'
        ]
        
        X = self.data[class_features]
        y = self.data['fracture_intensity']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest classifier
        self.classification_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=8, 
            random_state=42
        )
        self.classification_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.classification_model.predict(X_test)
        accuracy = self.classification_model.score(X_test, y_test)
        
        print(f"Classification Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.classification_model
    
    def train_refrac_model(self):
        """
        Train ML model for re-frac feasibility analysis
        """
        print("\nTraining re-frac feasibility model...")
        
        # Features for re-frac analysis
        refrac_features = [
            'porosity', 'permeability', 'fracture_half_length', 'production_decline',
            'reservoir_pressure_ratio', 'brittleness_index', 'toc', 'net_thickness'
        ]
        
        X = self.data[refrac_features]
        y = self.data['refrac_feasible']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest classifier
        self.refrac_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=8, 
            random_state=42
        )
        self.refrac_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.refrac_model.predict(X_test)
        accuracy = self.refrac_model.score(X_test, y_test)
        
        print(f"Re-frac Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Feasible', 'Feasible']))
        
        return self.refrac_model
    
    def optimize_fracture_design(self, reservoir_params):
        """
        Optimize fracture design for given reservoir parameters
        """
        print("\nOptimizing fracture design...")
        
        # Create parameter grid for optimization
        proppant_range = np.linspace(0.5, 4.0, 20)
        fluid_volume_range = np.linspace(2000, 10000, 20)
        
        best_fracture_length = 0
        best_params = {}
        
        for proppant in proppant_range:
            for fluid_vol in fluid_volume_range:
                # Create input vector
                input_params = reservoir_params.copy()
                input_params['proppant_concentration'] = proppant
                input_params['fluid_volume'] = fluid_vol
                
                # Calculate derived parameters
                input_params['brittleness_index'] = (
                    input_params['young_modulus'] / 1e6 + 
                    (1 - input_params['poisson_ratio']) * 20
                ) / 2
                input_params['stress_ratio'] = (
                    input_params['min_horizontal_stress'] / 
                    input_params['max_horizontal_stress']
                )
                
                # Prepare input for model
                opt_features = [
                    'porosity', 'permeability', 'young_modulus', 'poisson_ratio',
                    'min_horizontal_stress', 'max_horizontal_stress', 'net_thickness',
                    'toc', 'proppant_concentration', 'fluid_rate', 'fluid_volume',
                    'stage_length', 'brittleness_index', 'stress_ratio'
                ]
                
                X_input = np.array([[input_params[feature] for feature in opt_features]])
                X_input_scaled = self.scaler.transform(X_input)
                
                # Predict fracture length
                predicted_length = self.optimization_model.predict(X_input_scaled)[0]
                
                if predicted_length > best_fracture_length:
                    best_fracture_length = predicted_length
                    best_params = {
                        'proppant_concentration': proppant,
                        'fluid_volume': fluid_vol,
                        'predicted_fracture_length': predicted_length
                    }
        
        return best_params
    
    def predict_fracture_intensity(self, reservoir_params):
        """
        Predict fracture intensity classification
        """
        class_features = [
            'porosity', 'permeability', 'young_modulus', 'poisson_ratio',
            'brittleness_index', 'toc', 'stress_ratio', 'net_thickness'
        ]
        
        X_input = np.array([[reservoir_params[feature] for feature in class_features]])
        intensity = self.classification_model.predict(X_input)[0]
        probability = self.classification_model.predict_proba(X_input)[0].max()
        
        return intensity, probability
    
    def assess_refrac_feasibility(self, well_params):
        """
        Assess re-frac feasibility
        """
        refrac_features = [
            'porosity', 'permeability', 'fracture_half_length', 'production_decline',
            'reservoir_pressure_ratio', 'brittleness_index', 'toc', 'net_thickness'
        ]
        
        X_input = np.array([[well_params[feature] for feature in refrac_features]])
        feasible = self.refrac_model.predict(X_input)[0]
        probability = self.refrac_model.predict_proba(X_input)[0].max()
        
        return bool(feasible), probability
    
    def comprehensive_analysis(self, reservoir_params):
        """
        Perform comprehensive fracturing analysis
        """
        print("="*60)
        print("COMPREHENSIVE HYDRAULIC FRACTURING ANALYSIS")
        print("="*60)
        
        # Calculate derived parameters
        reservoir_params['brittleness_index'] = (
            reservoir_params['young_modulus'] / 1e6 + 
            (1 - reservoir_params['poisson_ratio']) * 20
        ) / 2
        reservoir_params['stress_ratio'] = (
            reservoir_params['min_horizontal_stress'] / 
            reservoir_params['max_horizontal_stress']
        )
        
        print("\nRESERVOIR PARAMETERS:")
        print(f"Porosity: {reservoir_params['porosity']:.3f}")
        print(f"Permeability: {reservoir_params['permeability']:.3f} mD")
        print(f"Young's Modulus: {reservoir_params['young_modulus']/1e6:.1f} MMpsi")
        print(f"Poisson's Ratio: {reservoir_params['poisson_ratio']:.3f}")
        print(f"Brittleness Index: {reservoir_params['brittleness_index']:.3f}")
        print(f"TOC: {reservoir_params['toc']:.1f}%")
        print(f"Net Thickness: {reservoir_params['net_thickness']:.1f} ft")
        
        # 1. Fracture Intensity Classification
        intensity, intensity_prob = self.predict_fracture_intensity(reservoir_params)
        print(f"\nFRACTURE INTENSITY: {intensity} (Confidence: {intensity_prob:.1%})")
        
        # 2. Fracture Design Optimization
        optimal_design = self.optimize_fracture_design(reservoir_params)
        print(f"\nOPTIMAL FRACTURE DESIGN:")
        print(f"Proppant Concentration: {optimal_design['proppant_concentration']:.2f} lb/gal")
        print(f"Fluid Volume: {optimal_design['fluid_volume']:,.0f} bbl")
        print(f"Predicted Fracture Half-Length: {optimal_design['predicted_fracture_length']:.1f} ft")
        
        # 3. Re-frac Feasibility (if production data available)
        if 'production_decline' in reservoir_params:
            reservoir_params['fracture_half_length'] = optimal_design['predicted_fracture_length']
            feasible, refrac_prob = self.assess_refrac_feasibility(reservoir_params)
            print(f"\nRE-FRAC FEASIBILITY: {'Feasible' if feasible else 'Not Feasible'} (Confidence: {refrac_prob:.1%})")
        
        print("="*60)
        
        return {
            'intensity': intensity,
            'optimal_design': optimal_design,
            'refrac_feasible': feasible if 'production_decline' in reservoir_params else None
        }

# Example usage and demonstration
def main():
    # Initialize the hydraulic fracturing optimization system
    frac_optimizer = HydraulicFracturingOptimization()
    
    # Generate synthetic data
    data = frac_optimizer.generate_synthetic_data(n_samples=1000)
    
    # Visualize the data
    frac_optimizer.visualize_data()
    
    # Train all models
    frac_optimizer.train_optimization_model()
    frac_optimizer.train_classification_model()
    frac_optimizer.train_refrac_model()
    
    # Example reservoir parameters for analysis
    example_reservoir = {
        'porosity': 0.08,
        'permeability': 0.5,  # mD
        'young_modulus': 5.2e6,  # psi
        'poisson_ratio': 0.22,
        'min_horizontal_stress': 5800,  # psi
        'max_horizontal_stress': 7200,  # psi
        'net_thickness': 180,  # ft
        'toc': 9.5,  # %
        'fluid_rate': 75,  # bbl/min
        'stage_length': 220,  # ft
        'production_decline': 0.35,
        'reservoir_pressure_ratio': 0.45
    }
    
    # Perform comprehensive analysis
    results = frac_optimizer.comprehensive_analysis(example_reservoir)
    
    print("\nProject completed successfully!")
    print("This system can be extended with:")
    print("- Real field data integration")
    print("- Advanced ML models (XGBoost, Neural Networks)")
    print("- Economic optimization")
    print("- Real-time monitoring integration")

if __name__ == "__main__":
    main()
