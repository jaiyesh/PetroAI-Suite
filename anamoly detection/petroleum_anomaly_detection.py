import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PetroleumDataGenerator:
    """Generate synthetic petroleum engineering data with anomalies"""
    
    def __init__(self, n_samples=10000, anomaly_fraction=0.05):
        self.n_samples = n_samples
        self.anomaly_fraction = anomaly_fraction
        self.n_anomalies = int(n_samples * anomaly_fraction)
        self.n_normal = n_samples - self.n_anomalies
    
    def generate_data(self):
        """Generate synthetic petroleum engineering data"""
        
        # Normal operating conditions
        # Pressure (psi) - typical range 2000-8000 psi
        pressure = np.random.normal(5000, 800, self.n_normal)
        pressure = np.clip(pressure, 2000, 8000)
        
        # Temperature (°F) - typical range 150-300°F
        temperature = np.random.normal(200, 30, self.n_normal)
        temperature = np.clip(temperature, 150, 300)
        
        # Flow rate (bbl/day) - correlated with pressure
        flow_rate = 50 + 0.8 * pressure + np.random.normal(0, 200, self.n_normal)
        flow_rate = np.clip(flow_rate, 100, 8000)
        
        # Gas-Oil Ratio (scf/bbl) - typical range 500-2000
        gor = np.random.normal(1000, 200, self.n_normal)
        gor = np.clip(gor, 500, 2000)
        
        # Water cut (%) - percentage of water in production
        water_cut = np.random.beta(2, 5, self.n_normal) * 100
        water_cut = np.clip(water_cut, 0, 80)
        
        # Wellhead pressure (psi) - correlated with reservoir pressure
        wellhead_pressure = pressure * 0.7 + np.random.normal(0, 100, self.n_normal)
        wellhead_pressure = np.clip(wellhead_pressure, 1000, 6000)
        
        # Choke size (64ths of an inch) - affects flow
        choke_size = np.random.choice([16, 20, 24, 32, 40, 48, 56, 64], self.n_normal)
        
        # Pump efficiency (%) - affects production
        pump_efficiency = np.random.normal(85, 10, self.n_normal)
        pump_efficiency = np.clip(pump_efficiency, 60, 100)
        
        # Generate anomalies
        # Anomaly 1: Equipment failure (low pump efficiency, abnormal pressure)
        anom1_count = self.n_anomalies // 3
        anom1_pressure = np.random.normal(3000, 500, anom1_count)
        anom1_temperature = np.random.normal(180, 20, anom1_count)
        anom1_flow_rate = np.random.normal(1000, 300, anom1_count)
        anom1_gor = np.random.normal(1500, 300, anom1_count)
        anom1_water_cut = np.random.normal(60, 15, anom1_count)
        anom1_wellhead_pressure = np.random.normal(2000, 300, anom1_count)
        anom1_choke_size = np.random.choice([16, 20, 24], anom1_count)
        anom1_pump_efficiency = np.random.normal(40, 10, anom1_count)
        
        # Anomaly 2: Reservoir issues (high water cut, changing GOR)
        anom2_count = self.n_anomalies // 3
        anom2_pressure = np.random.normal(4500, 600, anom2_count)
        anom2_temperature = np.random.normal(220, 25, anom2_count)
        anom2_flow_rate = np.random.normal(3000, 800, anom2_count)
        anom2_gor = np.random.normal(2500, 400, anom2_count)
        anom2_water_cut = np.random.normal(90, 5, anom2_count)
        anom2_wellhead_pressure = np.random.normal(3500, 400, anom2_count)
        anom2_choke_size = np.random.choice([32, 40, 48], anom2_count)
        anom2_pump_efficiency = np.random.normal(75, 12, anom2_count)
        
        # Anomaly 3: Extreme operating conditions
        anom3_count = self.n_anomalies - anom1_count - anom2_count
        anom3_pressure = np.random.normal(8500, 500, anom3_count)
        anom3_temperature = np.random.normal(350, 30, anom3_count)
        anom3_flow_rate = np.random.normal(9000, 1000, anom3_count)
        anom3_gor = np.random.normal(3000, 500, anom3_count)
        anom3_water_cut = np.random.normal(95, 3, anom3_count)
        anom3_wellhead_pressure = np.random.normal(7000, 300, anom3_count)
        anom3_choke_size = np.random.choice([56, 64], anom3_count)
        anom3_pump_efficiency = np.random.normal(95, 5, anom3_count)
        
        # Combine normal and anomalous data
        all_pressure = np.concatenate([pressure, anom1_pressure, anom2_pressure, anom3_pressure])
        all_temperature = np.concatenate([temperature, anom1_temperature, anom2_temperature, anom3_temperature])
        all_flow_rate = np.concatenate([flow_rate, anom1_flow_rate, anom2_flow_rate, anom3_flow_rate])
        all_gor = np.concatenate([gor, anom1_gor, anom2_gor, anom3_gor])
        all_water_cut = np.concatenate([water_cut, anom1_water_cut, anom2_water_cut, anom3_water_cut])
        all_wellhead_pressure = np.concatenate([wellhead_pressure, anom1_wellhead_pressure, anom2_wellhead_pressure, anom3_wellhead_pressure])
        all_choke_size = np.concatenate([choke_size, anom1_choke_size, anom2_choke_size, anom3_choke_size])
        all_pump_efficiency = np.concatenate([pump_efficiency, anom1_pump_efficiency, anom2_pump_efficiency, anom3_pump_efficiency])
        
        # Create labels (0 for normal, 1 for anomaly)
        labels = np.concatenate([np.zeros(self.n_normal), np.ones(self.n_anomalies)])
        
        # Create DataFrame
        data = pd.DataFrame({
            'pressure_psi': all_pressure,
            'temperature_f': all_temperature,
            'flow_rate_bbl_day': all_flow_rate,
            'gor_scf_bbl': all_gor,
            'water_cut_percent': all_water_cut,
            'wellhead_pressure_psi': all_wellhead_pressure,
            'choke_size_64ths': all_choke_size,
            'pump_efficiency_percent': all_pump_efficiency,
            'is_anomaly': labels
        })
        
        # Shuffle the data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return data

class AutoEncoder:
    """Autoencoder for anomaly detection"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self, encoding_dim=4):
        """Build the autoencoder model"""
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = Dense(16, activation='relu')(input_layer)
        encoded = Dense(8, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(8, activation='relu')(encoded)
        decoded = Dense(16, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder model
        self.model = Model(input_layer, decoded)
        self.model.compile(optimizer=Adam(learning_rate=0.001), 
                          loss='mse', 
                          metrics=['mae'])
        
        return self.model
    
    def train(self, X_train, epochs=100, batch_size=32, validation_split=0.2):
        """Train the autoencoder"""
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train only on normal data (assuming most data is normal)
        history = self.model.fit(X_train_scaled, X_train_scaled,
                               epochs=epochs,
                               batch_size=batch_size,
                               validation_split=validation_split,
                               shuffle=True,
                               verbose=1)
        
        return history
    
    def predict_anomalies(self, X_test, threshold_percentile=95):
        """Predict anomalies based on reconstruction error"""
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_test_scaled)
        
        # Calculate reconstruction error
        mse = np.mean(np.square(X_test_scaled - reconstructions), axis=1)
        
        # Set threshold based on percentile
        threshold = np.percentile(mse, threshold_percentile)
        
        # Predict anomalies
        anomalies = mse > threshold
        
        return anomalies, mse, threshold

class PetroleumAnomalyDetector:
    """Main class for petroleum engineering anomaly detection"""
    
    def __init__(self):
        self.data = None
        self.autoencoder = None
        self.isolation_forest = None
        self.scaler = StandardScaler()
        
    def load_data(self, data):
        """Load the petroleum engineering data"""
        self.data = data
        
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        # Separate features and labels
        X = self.data.drop('is_anomaly', axis=1)
        y = self.data['is_anomaly']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_isolation_forest(self, X_train, contamination=0.05):
        """Train Isolation Forest model"""
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.isolation_forest.fit(X_train_scaled)
        
        return self.isolation_forest
    
    def train_autoencoder(self, X_train):
        """Train AutoEncoder model"""
        # Initialize autoencoder
        self.autoencoder = AutoEncoder(input_dim=X_train.shape[1])
        self.autoencoder.build_model()
        
        # Train the model
        history = self.autoencoder.train(X_train, epochs=50, batch_size=32)
        
        return history
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate both models"""
        results = {}
        
        # Isolation Forest predictions
        X_test_scaled = self.scaler.transform(X_test)
        if_predictions = self.isolation_forest.predict(X_test_scaled)
        if_predictions = (if_predictions == -1).astype(int)  # Convert to 0/1
        
        # AutoEncoder predictions
        ae_predictions, mse, threshold = self.autoencoder.predict_anomalies(X_test)
        ae_predictions = ae_predictions.astype(int)
        
        # Store results
        results['isolation_forest'] = {
            'predictions': if_predictions,
            'classification_report': classification_report(y_test, if_predictions),
            'confusion_matrix': confusion_matrix(y_test, if_predictions)
        }
        
        results['autoencoder'] = {
            'predictions': ae_predictions,
            'reconstruction_errors': mse,
            'threshold': threshold,
            'classification_report': classification_report(y_test, ae_predictions),
            'confusion_matrix': confusion_matrix(y_test, ae_predictions)
        }
        
        return results
    
    def plot_results(self, X_test, y_test, results):
        """Plot the results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Data distribution
        axes[0, 0].scatter(X_test['pressure_psi'], X_test['flow_rate_bbl_day'], 
                          c=y_test, alpha=0.6, cmap='viridis')
        axes[0, 0].set_xlabel('Pressure (psi)')
        axes[0, 0].set_ylabel('Flow Rate (bbl/day)')
        axes[0, 0].set_title('Actual Anomalies')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Isolation Forest results
        axes[0, 1].scatter(X_test['pressure_psi'], X_test['flow_rate_bbl_day'], 
                          c=results['isolation_forest']['predictions'], alpha=0.6, cmap='viridis')
        axes[0, 1].set_xlabel('Pressure (psi)')
        axes[0, 1].set_ylabel('Flow Rate (bbl/day)')
        axes[0, 1].set_title('Isolation Forest Predictions')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: AutoEncoder results
        axes[0, 2].scatter(X_test['pressure_psi'], X_test['flow_rate_bbl_day'], 
                          c=results['autoencoder']['predictions'], alpha=0.6, cmap='viridis')
        axes[0, 2].set_xlabel('Pressure (psi)')
        axes[0, 2].set_ylabel('Flow Rate (bbl/day)')
        axes[0, 2].set_title('AutoEncoder Predictions')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Reconstruction errors
        axes[1, 0].hist(results['autoencoder']['reconstruction_errors'], bins=50, alpha=0.7)
        axes[1, 0].axvline(results['autoencoder']['threshold'], color='red', linestyle='--', 
                          label=f'Threshold: {results["autoencoder"]["threshold"]:.4f}')
        axes[1, 0].set_xlabel('Reconstruction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('AutoEncoder Reconstruction Errors')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Confusion Matrix - Isolation Forest
        cm_if = results['isolation_forest']['confusion_matrix']
        sns.heatmap(cm_if, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Isolation Forest Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        # Plot 6: Confusion Matrix - AutoEncoder
        cm_ae = results['autoencoder']['confusion_matrix']
        sns.heatmap(cm_ae, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2])
        axes[1, 2].set_title('AutoEncoder Confusion Matrix')
        axes[1, 2].set_xlabel('Predicted')
        axes[1, 2].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete anomaly detection analysis"""
        print("🛢️  PETROLEUM ENGINEERING ANOMALY DETECTION PROJECT")
        print("=" * 60)
        
        # Generate data
        print("\n1. Generating synthetic petroleum engineering data...")
        data_generator = PetroleumDataGenerator(n_samples=5000, anomaly_fraction=0.08)
        self.data = data_generator.generate_data()
        
        print(f"   Generated {len(self.data)} samples with {self.data['is_anomaly'].sum()} anomalies")
        print(f"   Features: {list(self.data.columns[:-1])}")
        
        # Display basic statistics
        print("\n2. Data Overview:")
        print(self.data.describe())
        
        # Preprocess data
        print("\n3. Preprocessing data...")
        X_train, X_test, y_train, y_test = self.preprocess_data()
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Train models
        print("\n4. Training Isolation Forest...")
        self.train_isolation_forest(X_train, contamination=0.08)
        print("   ✓ Isolation Forest trained successfully")
        
        print("\n5. Training AutoEncoder...")
        history = self.train_autoencoder(X_train)
        print("   ✓ AutoEncoder trained successfully")
        
        # Evaluate models
        print("\n6. Evaluating models...")
        results = self.evaluate_models(X_test, y_test)
        
        # Display results
        print("\n7. RESULTS:")
        print("\n" + "="*40)
        print("ISOLATION FOREST RESULTS:")
        print("="*40)
        print(results['isolation_forest']['classification_report'])
        
        print("\n" + "="*40)
        print("AUTOENCODER RESULTS:")
        print("="*40)
        print(results['autoencoder']['classification_report'])
        
        # Plot results
        print("\n8. Generating visualizations...")
        self.plot_results(X_test, y_test, results)
        
        # Feature importance analysis
        print("\n9. Feature Analysis:")
        feature_names = X_train.columns
        print(f"   Features analyzed: {list(feature_names)}")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        print("Both models have been successfully trained and evaluated for")
        print("petroleum engineering anomaly detection. The analysis includes:")
        print("• Equipment failure detection")
        print("• Reservoir performance monitoring")
        print("• Extreme operating condition identification")
        print("• Production optimization insights")
        
        return results

# Run the complete analysis
if __name__ == "__main__":
    detector = PetroleumAnomalyDetector()
    results = detector.run_complete_analysis()
