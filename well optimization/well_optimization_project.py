import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class WellPerformanceOptimizer:
    """
    A comprehensive well performance optimization system using machine learning
    for petroleum engineering applications.
    """
    
    def __init__(self, csv_file='well_data.csv'):
        self.csv_file = csv_file
        self.data = None
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        
    def load_data(self):
        """
        Load well performance data from CSV file
        """
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.data)} wells from {self.csv_file}")
            print(f"Columns: {list(self.data.columns)}")
            return self.data
        except FileNotFoundError:
            print(f"Error: {self.csv_file} not found!")
            print("Please run generate_well_dataset() first to create the data file.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def exploratory_data_analysis(self):
        """
        Perform comprehensive EDA on the well data
        """
        if self.data is None:
            self.load_data()
            if self.data is None:
                return None
        
        print("=== WELL PERFORMANCE DATA ANALYSIS ===\n")
        print(f"Dataset shape: {self.data.shape}")
        print(f"\nBasic Statistics:")
        print(self.data.describe().round(2))
        
        # Create visualizations
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Well Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Production Rate Distribution
        axes[0,0].hist(self.data['oil_rate_bbl_day'], bins=50, alpha=0.7, color='green')
        axes[0,0].set_title('Oil Production Rate Distribution')
        axes[0,0].set_xlabel('Oil Rate (bbl/day)')
        axes[0,0].set_ylabel('Frequency')
        
        # 2. Performance Index vs Key Parameters
        axes[0,1].scatter(self.data['permeability_md'], self.data['performance_index'], 
                         alpha=0.6, color='blue')
        axes[0,1].set_title('Performance vs Permeability')
        axes[0,1].set_xlabel('Permeability (mD)')
        axes[0,1].set_ylabel('Performance Index')
        axes[0,1].set_xscale('log')
        
        # 3. Water Cut Impact
        axes[0,2].scatter(self.data['water_cut_fraction'], self.data['oil_rate_bbl_day'], 
                         alpha=0.6, color='red')
        axes[0,2].set_title('Oil Rate vs Water Cut')
        axes[0,2].set_xlabel('Water Cut (fraction)')
        axes[0,2].set_ylabel('Oil Rate (bbl/day)')
        
        # 4. Pressure Analysis
        axes[1,0].scatter(self.data['reservoir_pressure_psi'] - self.data['bottomhole_pressure_psi'], 
                         self.data['oil_rate_bbl_day'], alpha=0.6, color='purple')
        axes[1,0].set_title('Oil Rate vs Pressure Drawdown')
        axes[1,0].set_xlabel('Pressure Drawdown (psi)')
        axes[1,0].set_ylabel('Oil Rate (bbl/day)')
        
        # 5. Economic Performance
        axes[1,1].scatter(self.data['daily_revenue_usd'], self.data['performance_index'], 
                         alpha=0.6, color='orange')
        axes[1,1].set_title('Performance vs Daily Revenue')
        axes[1,1].set_xlabel('Daily Revenue ($)')
        axes[1,1].set_ylabel('Performance Index')
        
        # 6. Choke Size Optimization
        choke_groups = self.data.groupby(pd.cut(self.data['choke_size_64th'], bins=8))['oil_rate_bbl_day'].mean()
        axes[1,2].bar(range(len(choke_groups)), choke_groups.values, color='brown', alpha=0.7)
        axes[1,2].set_title('Average Oil Rate by Choke Size')
        axes[1,2].set_xlabel('Choke Size Groups')
        axes[1,2].set_ylabel('Average Oil Rate (bbl/day)')
        
        # 7. Correlation Heatmap
        corr_cols = ['permeability_md', 'porosity_fraction', 'choke_size_64th', 
                     'reservoir_pressure_psi', 'water_cut_fraction', 'oil_rate_bbl_day', 'performance_index']
        corr_matrix = self.data[corr_cols].corr()
        im = axes[2,0].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[2,0].set_xticks(range(len(corr_cols)))
        axes[2,0].set_yticks(range(len(corr_cols)))
        axes[2,0].set_xticklabels([col.replace('_', '\n') for col in corr_cols], rotation=45, ha='right')
        axes[2,0].set_yticklabels([col.replace('_', '\n') for col in corr_cols])
        axes[2,0].set_title('Correlation Matrix')
        plt.colorbar(im, ax=axes[2,0])
        
        # 8. Performance Distribution
        axes[2,1].hist(self.data['performance_index'], bins=50, alpha=0.7, color='gold')
        axes[2,1].set_title('Performance Index Distribution')
        axes[2,1].set_xlabel('Performance Index')
        axes[2,1].set_ylabel('Frequency')
        
        # 9. Top vs Bottom Performers
        top_performers = self.data.nlargest(50, 'performance_index')
        bottom_performers = self.data.nsmallest(50, 'performance_index')
        
        metrics = ['permeability_md', 'water_cut_fraction', 'choke_size_64th']
        x_pos = np.arange(len(metrics))
        
        top_means = [top_performers[metric].mean() for metric in metrics]
        bottom_means = [bottom_performers[metric].mean() for metric in metrics]
        
        # Normalize for comparison
        top_norm = [(val - self.data[metrics[i]].min()) / (self.data[metrics[i]].max() - self.data[metrics[i]].min()) 
                   for i, val in enumerate(top_means)]
        bottom_norm = [(val - self.data[metrics[i]].min()) / (self.data[metrics[i]].max() - self.data[metrics[i]].min()) 
                      for i, val in enumerate(bottom_means)]
        
        width = 0.35
        axes[2,2].bar(x_pos - width/2, top_norm, width, label='Top 50 Wells', color='green', alpha=0.7)
        axes[2,2].bar(x_pos + width/2, bottom_norm, width, label='Bottom 50 Wells', color='red', alpha=0.7)
        axes[2,2].set_title('Top vs Bottom Performers (Normalized)')
        axes[2,2].set_xlabel('Parameters')
        axes[2,2].set_ylabel('Normalized Value')
        axes[2,2].set_xticks(x_pos)
        axes[2,2].set_xticklabels([m.replace('_', '\n') for m in metrics])
        axes[2,2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Key insights
        print("\n=== KEY INSIGHTS ===")
        print(f"Average oil production: {self.data['oil_rate_bbl_day'].mean():.1f} bbl/day")
        print(f"Average water cut: {self.data['water_cut_fraction'].mean():.2%}")
        print(f"Average performance index: {self.data['performance_index'].mean():.1f}")
        print(f"Best performing well: {self.data['performance_index'].max():.1f}")
        print(f"Worst performing well: {self.data['performance_index'].min():.1f}")
        
        return self.data
    
    def prepare_ml_features(self):
        """
        Prepare features for machine learning models
        """
        if self.data is None:
            self.load_data()
            if self.data is None:
                return None, None, None
        
        # Feature engineering
        self.data['pressure_drawdown'] = self.data['reservoir_pressure_psi'] - self.data['bottomhole_pressure_psi']
        self.data['productivity_factor'] = self.data['permeability_md'] * self.data['porosity_fraction']
        self.data['economic_efficiency'] = self.data['daily_revenue_usd'] / self.data['daily_opex_usd']
        self.data['total_fluid_rate'] = self.data['oil_rate_bbl_day'] + self.data['water_rate_bbl_day']
        
        # Select features for modeling
        feature_columns = [
            'permeability_md', 'porosity_fraction', 'well_depth_ft', 'tubing_diameter_in',
            'choke_size_64th', 'reservoir_pressure_psi', 'oil_gravity_api', 'gas_oil_ratio_scf_bbl',
            'bottomhole_pressure_psi', 'water_cut_fraction', 'pressure_drawdown', 'productivity_factor'
        ]
        
        X = self.data[feature_columns].copy()
        y = self.data['performance_index'].copy()
        
        return X, y, feature_columns
    
    def train_models(self):
        """
        Train multiple ML models for well performance optimization
        """
        X, y, feature_columns = self.prepare_ml_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        print("=== MODEL TRAINING RESULTS ===\n")
        
        for name, model in models.items():
            # Train model
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"{name}:")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.3f}")
            print()
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        self.best_model = results[best_model_name]['model']
        self.models = results
        
        print(f"Best model: {best_model_name} (R² = {results[best_model_name]['r2']:.3f})")
        
        # Feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 5 Most Important Features ({best_model_name}):")
            for i, row in importance_df.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Plot results
        self._plot_model_results(X_test, y_test, feature_columns)
        
        return results
    
    def _plot_model_results(self, X_test, y_test, feature_columns):
        """
        Plot model training results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Machine Learning Model Results', fontsize=16, fontweight='bold')
        
        # Model comparison
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2'] for name in model_names]
        rmse_scores = [self.models[name]['rmse'] for name in model_names]
        
        axes[0,0].bar(model_names, r2_scores, color=['blue', 'green', 'orange'], alpha=0.7)
        axes[0,0].set_title('Model R² Comparison')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_ylim(0, 1)
        
        axes[0,1].bar(model_names, rmse_scores, color=['blue', 'green', 'orange'], alpha=0.7)
        axes[0,1].set_title('Model RMSE Comparison')
        axes[0,1].set_ylabel('RMSE')
        
        # Best model predictions vs actual
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        best_predictions = self.models[best_model_name]['predictions']
        
        axes[1,0].scatter(y_test, best_predictions, alpha=0.6, color='red')
        axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        axes[1,0].set_xlabel('Actual Performance Index')
        axes[1,0].set_ylabel('Predicted Performance Index')
        axes[1,0].set_title(f'Predictions vs Actual ({best_model_name})')
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            axes[1,1].barh(range(len(importance_df)), importance_df['importance'], color='purple', alpha=0.7)
            axes[1,1].set_yticks(range(len(importance_df)))
            axes[1,1].set_yticklabels([f.replace('_', ' ').title() for f in importance_df['feature']])
            axes[1,1].set_xlabel('Feature Importance')
            axes[1,1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
    
    def optimize_well_parameters(self, target_wells=None):
        """
        Optimize well parameters for maximum performance
        """
        if self.best_model is None:
            self.train_models()
        
        X, y, feature_columns = self.prepare_ml_features()
        
        if target_wells is None:
            # Select bottom 10% performers for optimization
            n_optimize = max(10, int(0.1 * len(self.data)))
            target_wells = self.data.nsmallest(n_optimize, 'performance_index')['well_id'].values
        
        optimization_results = []
        
        print("=== WELL OPTIMIZATION RECOMMENDATIONS ===\n")
        
        for well_id in target_wells[:5]:  # Show first 5 for brevity
            well_data = self.data[self.data['well_id'] == well_id].iloc[0]
            current_performance = well_data['performance_index']
            
            # Current parameters
            current_params = well_data[feature_columns].values.reshape(1, -1)
            
            # Optimization scenarios
            scenarios = self._generate_optimization_scenarios(well_data, feature_columns)
            
            best_scenario = None
            best_improvement = 0
            
            for scenario_name, scenario_params in scenarios.items():
                if hasattr(self.best_model, 'predict'):
                    if 'Linear' in str(type(self.best_model)):
                        predicted_performance = self.best_model.predict(
                            self.scaler.transform(scenario_params.reshape(1, -1))
                        )[0]
                    else:
                        predicted_performance = self.best_model.predict(scenario_params.reshape(1, -1))[0]
                    
                    improvement = predicted_performance - current_performance
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_scenario = (scenario_name, scenario_params, predicted_performance)
            
            if best_scenario:
                scenario_name, optimized_params, predicted_performance = best_scenario
                
                print(f"Well {well_id}:")
                print(f"  Current Performance: {current_performance:.1f}")
                print(f"  Optimized Performance: {predicted_performance:.1f}")
                print(f"  Improvement: {best_improvement:.1f} ({best_improvement/current_performance*100:.1f}%)")
                print(f"  Best Scenario: {scenario_name}")
                print()
                
                optimization_results.append({
                    'well_id': well_id,
                    'current_performance': current_performance,
                    'optimized_performance': predicted_performance,
                    'improvement': best_improvement,
                    'scenario': scenario_name
                })
        
        return optimization_results
    
    def _generate_optimization_scenarios(self, well_data, feature_columns):
        """
        Generate optimization scenarios for a well
        """
        scenarios = {}
        base_params = well_data[feature_columns].values
        
        # Scenario 1: Optimize choke size
        choke_optimized = base_params.copy()
        choke_idx = feature_columns.index('choke_size_64th')
        choke_optimized[choke_idx] = min(64, choke_optimized[choke_idx] * 1.5)
        scenarios['Choke Size Optimization'] = choke_optimized
        
        # Scenario 2: Reduce water cut (workover)
        water_cut_optimized = base_params.copy()
        water_cut_idx = feature_columns.index('water_cut_fraction')
        water_cut_optimized[water_cut_idx] = max(0.05, water_cut_optimized[water_cut_idx] * 0.7)
        scenarios['Water Cut Reduction'] = water_cut_optimized
        
        # Scenario 3: Pressure maintenance
        pressure_optimized = base_params.copy()
        bhp_idx = feature_columns.index('bottomhole_pressure_psi')
        res_press_idx = feature_columns.index('reservoir_pressure_psi')
        pressure_optimized[bhp_idx] = min(pressure_optimized[res_press_idx] - 100, 
                                        pressure_optimized[bhp_idx] * 1.2)
        scenarios['Pressure Optimization'] = pressure_optimized
        
        # Scenario 4: Combined optimization
        combined_optimized = base_params.copy()
        combined_optimized[choke_idx] = min(64, combined_optimized[choke_idx] * 1.3)
        combined_optimized[water_cut_idx] = max(0.05, combined_optimized[water_cut_idx] * 0.8)
        combined_optimized[bhp_idx] = min(combined_optimized[res_press_idx] - 100, 
                                        combined_optimized[bhp_idx] * 1.1)
        scenarios['Combined Optimization'] = combined_optimized
        
        return scenarios
    
    def economic_analysis(self):
        """
        Perform detailed economic analysis with profitability insights
        """
        if self.data is None:
            self.load_data()
            if self.data is None:
                return None, None
        
        # Calculate key economic metrics
        profitable_wells = self.data[self.data['daily_revenue_usd'] > 0]
        marginal_wells = self.data[(self.data['daily_revenue_usd'] >= -50) & 
                                  (self.data['daily_revenue_usd'] <= 100)]
        
        print("=== ECONOMIC ANALYSIS ===\n")
        print(f"Profitable Wells: {len(profitable_wells)} ({len(profitable_wells)/len(self.data)*100:.1f}%)")
        print(f"Marginal Wells: {len(marginal_wells)} ({len(marginal_wells)/len(self.data)*100:.1f}%)")
        print(f"Average Daily Revenue: ${self.data['daily_revenue_usd'].mean():.0f}")
        print(f"Average Daily OPEX: ${self.data['daily_opex_usd'].mean():.0f}")
        
        # Breakeven analysis
        breakeven_rate = self.data['daily_opex_usd'] / (self.data['oil_price_usd_bbl'] + 
                        self.data['gas_rate_scf_day'] * self.data['gas_price_usd_mcf'] / 
                        (1000 * self.data['oil_rate_bbl_day'] + 0.001))
        
        print(f"\nAverage Breakeven Oil Rate: {breakeven_rate.mean():.1f} bbl/day")
        
        # Economic optimization potential
        underperforming = self.data[self.data['daily_revenue_usd'] < 0]
        if len(underperforming) > 0:
            avg_loss = underperforming['daily_revenue_usd'].mean()
            total_annual_loss = len(underperforming) * avg_loss * 365
            print(f"Wells Operating at Loss: {len(underperforming)}")
            print(f"Average Daily Loss: ${abs(avg_loss):.0f}")
            print(f"Total Annual Loss: ${abs(total_annual_loss):,.0f}")
            
            # Optimization potential
            if len(underperforming) > 0:
                # Assume 30% improvement is achievable
                potential_savings = abs(total_annual_loss) * 0.3
                print(f"Potential Annual Savings (30% improvement): ${potential_savings:,.0f}")
        
        return profitable_wells, marginal_wells
    
    def generate_report(self):
        """
        Generate a comprehensive optimization report
        """
        if self.data is None:
            self.load_data()
            if self.data is None:
                return None
        
        if not self.models:
            self.train_models()
        
        print("="*60)
        print("WELL PERFORMANCE OPTIMIZATION REPORT")
        print("="*60)
        
        print(f"\nDATA SUMMARY:")
        print(f"Total Wells Analyzed: {len(self.data)}")
        print(f"Average Oil Production: {self.data['oil_rate_bbl_day'].mean():.1f} bbl/day")
        print(f"Production Range: {self.data['oil_rate_bbl_day'].min():.1f} - {self.data['oil_rate_bbl_day'].max():.1f} bbl/day")
        print(f"Average Performance Index: {self.data['performance_index'].mean():.1f}")
        print(f"Performance Range: {self.data['performance_index'].min():.1f} - {self.data['performance_index'].max():.1f}")
        
        # Economic summary
        profitable_wells = len(self.data[self.data['daily_revenue_usd'] > 0])
        print(f"Profitable Wells: {profitable_wells} ({profitable_wells/len(self.data)*100:.1f}%)")
        
        print(f"\nMODEL PERFORMANCE:")
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        best_r2 = self.models[best_model_name]['r2']
        print(f"Best Model: {best_model_name}")
        print(f"Model Accuracy (R²): {best_r2:.3f}")
        print(f"Model RMSE: {self.models[best_model_name]['rmse']:.2f}")
        
        # Economic impact assessment
        low_performers = self.data[self.data['performance_index'] < self.data['performance_index'].quantile(0.25)]
        potential_improvement = len(low_performers) * 100 * 365  # More realistic improvement
        
        print(f"\nECONOMIC OPPORTUNITY:")
        print(f"Wells Below 25th Percentile: {len(low_performers)}")
        print(f"Potential Annual Revenue Increase: ${potential_improvement:,.0f}")
        
        # Production efficiency analysis
        high_performers = self.data[self.data['performance_index'] > self.data['performance_index'].quantile(0.75)]
        print(f"\nPERFORMANCE INSIGHTS:")
        print(f"Top Quartile Wells Average Production: {high_performers['oil_rate_bbl_day'].mean():.1f} bbl/day")
        print(f"Bottom Quartile Wells Average Production: {low_performers['oil_rate_bbl_day'].mean():.1f} bbl/day")
        print(f"Production Improvement Potential: {high_performers['oil_rate_bbl_day'].mean() - low_performers['oil_rate_bbl_day'].mean():.1f} bbl/day")
        
        print(f"\nRECOMMENDATIONS:")
        print("1. Focus on choke size optimization for immediate gains")
        print("2. Implement water cut reduction programs for high water-cut wells")
        print("3. Consider pressure maintenance in depleted reservoirs")
        print("4. Regular performance monitoring using the ML model")
        print("5. Prioritize optimization of wells with negative cash flow")
        
        return {
            'total_wells': len(self.data),
            'profitable_wells': profitable_wells,
            'model_accuracy': best_r2,
            'economic_opportunity': potential_improvement,
            'low_performers': len(low_performers),
            'avg_production': self.data['oil_rate_bbl_day'].mean()
        }
    
    def field_development_strategy(self):
        """
        Develop field-wide optimization strategy
        """
        if self.data is None:
            self.load_data()
            if self.data is None:
                return None
        
        print("=== FIELD DEVELOPMENT STRATEGY ===\n")
        
        # Segment wells by profitability
        profitable = self.data[self.data['daily_revenue_usd'] > 100]
        marginal = self.data[(self.data['daily_revenue_usd'] >= 0) & 
                            (self.data['daily_revenue_usd'] <= 100)]
        unprofitable = self.data[self.data['daily_revenue_usd'] < 0]
        
        print(f"Well Classification:")
        print(f"  Profitable Wells (>$100/day): {len(profitable)} wells")
        print(f"  Marginal Wells ($0-100/day): {len(marginal)} wells") 
        print(f"  Unprofitable Wells (<$0/day): {len(unprofitable)} wells")
        
        # Development recommendations by segment
        print(f"\nDevelopment Recommendations:")
        
        if len(profitable) > 0:
            print(f"  PROFITABLE WELLS:")
            print(f"    - Maintain current operations")
            print(f"    - Consider infill drilling in similar areas")
            print(f"    - Average characteristics: {profitable['permeability_md'].mean():.0f} mD perm, {profitable['water_cut_fraction'].mean():.1%} water cut")
        
        if len(marginal) > 0:
            print(f"  MARGINAL WELLS:")
            print(f"    - Implement cost reduction strategies")
            print(f"    - Optimize artificial lift systems")
            print(f"    - Consider workover operations")
        
        if len(unprofitable) > 0:
            print(f"  UNPROFITABLE WELLS:")
            print(f"    - Priority for optimization or abandonment")
            print(f"    - Evaluate for enhanced oil recovery")
            print(f"    - Consider plugging if optimization fails")
        
        # Sweet spot identification
        sweet_spot = self.data[
            (self.data['permeability_md'] > self.data['permeability_md'].quantile(0.6)) &
            (self.data['water_cut_fraction'] < self.data['water_cut_fraction'].quantile(0.4)) &
            (self.data['performance_index'] > self.data['performance_index'].quantile(0.7))
        ]
        
        print(f"\nSWEET SPOT IDENTIFICATION:")
        print(f"  High-quality locations identified: {len(sweet_spot)} areas")
        if len(sweet_spot) > 0:
            print(f"  Average performance: {sweet_spot['performance_index'].mean():.1f}")
            print(f"  Average production: {sweet_spot['oil_rate_bbl_day'].mean():.1f} bbl/day")
            print(f"  Recommended for future development")
        
        return {
            'profitable_count': len(profitable),
            'marginal_count': len(marginal),
            'unprofitable_count': len(unprofitable),
            'sweet_spot_count': len(sweet_spot)
        }

# Data Generation Function (Run once to create dataset)
def generate_well_dataset(n_wells=500, filename='well_data.csv'):
    """
    Generate realistic well performance data and save to CSV file
    This function should be run once to create the dataset
    """
    print(f"Generating {n_wells} wells dataset...")
    np.random.seed(42)
    
    # Reservoir Properties
    permeability = np.random.lognormal(mean=2, sigma=1, size=n_wells)  # mD
    porosity = np.random.normal(0.15, 0.05, n_wells)  # fraction
    porosity = np.clip(porosity, 0.05, 0.35)
    
    # Well Design Parameters
    well_depth = np.random.normal(8000, 1500, n_wells)  # ft
    well_depth = np.clip(well_depth, 4000, 15000)
    
    tubing_diameter = np.random.choice([2.875, 3.5, 4.5, 5.5], n_wells)  # inches
    choke_size = np.random.uniform(8, 64, n_wells)  # 64ths of inch
    
    # Reservoir Pressure and Temperature
    reservoir_pressure = np.random.normal(3500, 800, n_wells)  # psi
    reservoir_pressure = np.clip(reservoir_pressure, 1500, 6000)
    
    reservoir_temp = 150 + well_depth * 0.015  # °F (geothermal gradient)
    
    # Fluid Properties
    oil_gravity = np.random.normal(35, 8, n_wells)  # API
    oil_gravity = np.clip(oil_gravity, 15, 50)
    
    gas_oil_ratio = np.random.lognormal(5, 0.8, n_wells)  # scf/bbl
    
    # Operating Conditions
    bottomhole_pressure = reservoir_pressure - np.random.uniform(200, 800, n_wells)
    wellhead_pressure = bottomhole_pressure - np.random.uniform(100, 500, n_wells)
    wellhead_pressure = np.clip(wellhead_pressure, 50, bottomhole_pressure)
    
    # Water Cut (increases with time/depletion)
    water_cut = np.random.beta(2, 8, n_wells) * 0.8  # fraction
    
    # Calculate Oil Production Rate using simplified IPR (Inflow Performance Relationship)
    # Based on Vogel's equation and Darcy's law principles
    productivity_index = (permeability * porosity * well_depth) / (1000 * oil_gravity)
    pressure_drawdown = reservoir_pressure - bottomhole_pressure
    
    # Base oil rate calculation - improved scaling
    oil_rate = productivity_index * pressure_drawdown * (1 - water_cut) / 20  # Better scaling
    
    # Apply choke effect
    choke_factor = np.minimum(choke_size / 32, 1.0)
    oil_rate *= choke_factor
    
    # Apply well completion efficiency factor
    completion_efficiency = np.random.uniform(0.6, 0.95, n_wells)
    oil_rate *= completion_efficiency
    
    # Add some noise and ensure realistic ranges
    oil_rate += np.random.normal(0, oil_rate * 0.1)
    oil_rate = np.clip(oil_rate, 5, 1500)  # bbl/day - more realistic range
    
    # Gas Production Rate
    gas_rate = oil_rate * gas_oil_ratio  # scf/day
    
    # Water Production Rate
    water_rate = oil_rate * water_cut / (1 - water_cut + 1e-6)
    water_rate = np.clip(water_rate, 0, oil_rate * 5)
    
    # Economic Parameters
    oil_price = np.random.normal(75, 15, n_wells)  # $/bbl
    gas_price = np.random.normal(3.5, 0.8, n_wells)  # $/Mcf
    
    # Operating Costs - more realistic calculation
    base_daily_opex = 150 + well_depth * 0.01  # Base cost per well
    variable_opex = oil_rate * 2.5 + water_rate * 1.0  # Variable costs
    daily_opex = base_daily_opex + variable_opex  # $/day
    
    # Revenue Calculation
    daily_revenue = (oil_rate * oil_price + 
                    gas_rate * gas_price / 1000 - 
                    daily_opex)
    
    # Well Performance Index (target variable)
    # Combines production efficiency and economic performance
    performance_index = (oil_rate * (1 - water_cut) * oil_price / daily_opex) * 100
    
    # Create DataFrame
    data = pd.DataFrame({
        'well_id': range(1, n_wells + 1),
        'permeability_md': permeability,
        'porosity_fraction': porosity,
        'well_depth_ft': well_depth,
        'tubing_diameter_in': tubing_diameter,
        'choke_size_64th': choke_size,
        'reservoir_pressure_psi': reservoir_pressure,
        'reservoir_temp_f': reservoir_temp,
        'oil_gravity_api': oil_gravity,
        'gas_oil_ratio_scf_bbl': gas_oil_ratio,
        'bottomhole_pressure_psi': bottomhole_pressure,
        'wellhead_pressure_psi': wellhead_pressure,
        'water_cut_fraction': water_cut,
        'oil_rate_bbl_day': oil_rate,
        'gas_rate_scf_day': gas_rate,
        'water_rate_bbl_day': water_rate,
        'oil_price_usd_bbl': oil_price,
        'gas_price_usd_mcf': gas_price,
        'daily_opex_usd': daily_opex,
        'daily_revenue_usd': daily_revenue,
        'performance_index': performance_index
    })
    
    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Display basic statistics
    print(f"\nDataset Summary:")
    print(f"Average Oil Production: {data['oil_rate_bbl_day'].mean():.1f} bbl/day")
    print(f"Average Performance Index: {data['performance_index'].mean():.1f}")
    print(f"Profitable Wells: {len(data[data['daily_revenue_usd'] > 0])} ({len(data[data['daily_revenue_usd'] > 0])/len(data)*100:.1f}%)")
    
    return data

# Additional utility functions
def create_optimization_dashboard(optimizer):
    """
    Create an interactive optimization dashboard
    """
    if optimizer.data is None or not optimizer.models:
        print("Please run the complete analysis first!")
        return
    
    # Performance vs Investment Analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Well Optimization Dashboard', fontsize=16, fontweight='bold')
    
    # 1. ROI Analysis
    roi = (optimizer.data['daily_revenue_usd'] * 365) / (optimizer.data['daily_opex_usd'] * 365 + 50000)  # Assume $50k capex
    axes[0,0].scatter(optimizer.data['oil_rate_bbl_day'], roi, alpha=0.6, c=optimizer.data['performance_index'], cmap='viridis')
    axes[0,0].set_xlabel('Oil Rate (bbl/day)')
    axes[0,0].set_ylabel('ROI')
    axes[0,0].set_title('ROI vs Production Rate')
    
    # 2. Optimization Priority Matrix
    axes[0,1].scatter(optimizer.data['performance_index'], optimizer.data['daily_revenue_usd'], 
                     alpha=0.6, s=optimizer.data['oil_rate_bbl_day']*2)
    axes[0,1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0,1].set_xlabel('Performance Index')
    axes[0,1].set_ylabel('Daily Revenue ($)')
    axes[0,1].set_title('Optimization Priority Matrix')
    
    # 3. Water Cut Impact
    water_cut_bins = pd.cut(optimizer.data['water_cut_fraction'], bins=5)
    water_cut_impact = optimizer.data.groupby(water_cut_bins)['oil_rate_bbl_day'].mean()
    axes[0,2].bar(range(len(water_cut_impact)), water_cut_impact.values, alpha=0.7)
    axes[0,2].set_title('Production vs Water Cut')
    axes[0,2].set_xlabel('Water Cut Level')
    axes[0,2].set_ylabel('Average Oil Rate (bbl/day)')
    
    # 4. Economic Sensitivity
    oil_prices = np.arange(50, 101, 10)
    sensitivities = []
    for price in oil_prices:
        temp_revenue = (optimizer.data['oil_rate_bbl_day'] * price + 
                       optimizer.data['gas_rate_scf_day'] * 3.5 / 1000 - 
                       optimizer.data['daily_opex_usd'])
        profitable_wells = len(temp_revenue[temp_revenue > 0])
        sensitivities.append(profitable_wells)
    
    axes[1,0].plot(oil_prices, sensitivities, 'o-', linewidth=2, markersize=6)
    axes[1,0].set_xlabel('Oil Price ($/bbl)')
    axes[1,0].set_ylabel('Profitable Wells Count')
    axes[1,0].set_title('Economic Sensitivity to Oil Price')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Reservoir Quality vs Performance
    axes[1,1].scatter(optimizer.data['permeability_md'], optimizer.data['porosity_fraction'], 
                     c=optimizer.data['performance_index'], s=50, alpha=0.7, cmap='coolwarm')
    axes[1,1].set_xlabel('Permeability (mD)')
    axes[1,1].set_ylabel('Porosity (fraction)')
    axes[1,1].set_title('Reservoir Quality Map')
    axes[1,1].set_xscale('log')
    
    # 6. Optimization Potential
    current_performance = optimizer.data['performance_index']
    potential_performance = current_performance * 1.3  # Assume 30% improvement potential
    improvement = potential_performance - current_performance
    
    # Add improvement as a temporary column to get top candidates
    temp_data = optimizer.data.copy()
    temp_data['improvement_potential'] = improvement
    top_candidates = temp_data.nlargest(20, 'improvement_potential')
    top_improvement_values = improvement.nlargest(20)
    
    axes[1,2].barh(range(len(top_improvement_values)), top_improvement_values.values)
    axes[1,2].set_xlabel('Performance Improvement Potential')
    axes[1,2].set_ylabel('Well Rank')
    axes[1,2].set_title('Top 20 Optimization Candidates')
    
    plt.tight_layout()
    plt.show()
    
    return roi, improvement

# Example usage and demonstration
def main():
    """
    Main execution function to demonstrate the well optimization system
    """
    print("Initializing Well Performance Optimization System...")
    
    # Check if data file exists, if not generate it
    import os
    data_file = 'well_data.csv'
    
    if not os.path.exists(data_file):
        print(f"\nData file '{data_file}' not found. Generating new dataset...")
        generate_well_dataset(n_wells=500, filename=data_file)
    else:
        print(f"\nUsing existing data file: {data_file}")
    
    # Initialize optimizer with CSV file
    optimizer = WellPerformanceOptimizer(csv_file=data_file)
    
    # Load and analyze data
    print("\n1. Loading well data from CSV...")
    if optimizer.load_data() is None:
        print("Failed to load data. Exiting...")
        return None, None
    
    print("\n2. Performing exploratory data analysis...")
    optimizer.exploratory_data_analysis()
    
    print("\n3. Training machine learning models...")
    optimizer.train_models()
    
    print("\n4. Performing economic analysis...")
    optimizer.economic_analysis()
    
    print("\n5. Optimizing well parameters...")
    optimization_results = optimizer.optimize_well_parameters()
    
    print("\n6. Generating comprehensive report...")
    report = optimizer.generate_report()
    
    print("\n7. Developing field strategy...")
    field_strategy = optimizer.field_development_strategy()
    
    print("\n8. Creating optimization dashboard...")
    roi, improvement = create_optimization_dashboard(optimizer)
    
    return optimizer, report

if __name__ == "__main__":
    # Option 1: Run the complete analysis (will generate data if needed)
    optimizer, report = main()
    
    # Option 2: Generate new dataset only (uncomment to use)
    # generate_well_dataset(n_wells=1000, filename='well_data_large.csv')
    
    # Option 3: Use custom data file (uncomment to use)
    # optimizer = WellPerformanceOptimizer(csv_file='your_custom_data.csv')
    # optimizer, report = main()
    
    # Additional analysis examples
    if optimizer is not None and optimizer.data is not None:
        print("\n" + "="*60)
        print("ADDITIONAL ANALYSIS CAPABILITIES")
        print("="*60)
        
        # Example: Analyze specific well type
        high_perm_wells = optimizer.data[optimizer.data['permeability_md'] > 100]
        print(f"\nHigh Permeability Wells (>100 mD): {len(high_perm_wells)}")
        if len(high_perm_wells) > 0:
            print(f"Average Performance: {high_perm_wells['performance_index'].mean():.1f}")
        
        # Example: Economic scenarios
        oil_price_scenarios = [60, 75, 90]  # $/bbl
        print(f"\nEconomic Sensitivity Analysis:")
        for price in oil_price_scenarios:
            temp_revenue = (optimizer.data['oil_rate_bbl_day'] * price + 
                           optimizer.data['gas_rate_scf_day'] * 3.5 / 1000 - 
                           optimizer.data['daily_opex_usd'])
            avg_revenue = temp_revenue.mean()
            print(f"  Oil @ ${price}/bbl: Average daily revenue = ${avg_revenue:.0f}")
        
        print(f"\nProject completed successfully!")
        print(f"Data saved in: well_data.csv")
        print(f"Use optimizer.data to access the dataset")
        print(f"Use optimizer.best_model to make predictions on new wells")