"""
Well Performance Data Preparation Module
========================================

This module handles all data generation and preparation tasks for the 
Well Performance Optimization project. It creates realistic synthetic 
well data based on petroleum engineering principles.

Author: Petroleum Engineering & Data Science Team
Date: 2025
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WellDataGenerator:
    """
    A comprehensive data generator for realistic well performance data
    based on petroleum engineering principles and field observations.
    """
    
    def __init__(self, random_seed=42):
        """
        Initialize the data generator with reproducible random seed
        
        Parameters:
        -----------
        random_seed : int
            Random seed for reproducible results
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_reservoir_properties(self, n_wells):
        """
        Generate realistic reservoir properties based on geological distributions
        
        Parameters:
        -----------
        n_wells : int
            Number of wells to generate
            
        Returns:
        --------
        dict : Dictionary containing reservoir properties
        """
        print("Generating reservoir properties...")
        
        # Permeability (mD) - Lognormal distribution typical for reservoir rocks
        # Mean of 2 in log space ≈ 7.4 mD, which is reasonable for many reservoirs
        permeability = np.random.lognormal(mean=2, sigma=1, size=n_wells)
        
        # Porosity (fraction) - Normal distribution, typical range 5-35%
        porosity = np.random.normal(0.15, 0.05, n_wells)
        porosity = np.clip(porosity, 0.05, 0.35)
        
        # Net-to-Gross ratio - fraction of reservoir rock that's productive
        net_to_gross = np.random.beta(8, 3, n_wells)  # Skewed toward higher values
        net_to_gross = np.clip(net_to_gross, 0.3, 1.0)
        
        # Reservoir thickness (ft)
        thickness = np.random.gamma(2, 25, n_wells)  # Gamma distribution
        thickness = np.clip(thickness, 10, 200)
        
        return {
            'permeability_md': permeability,
            'porosity_fraction': porosity,
            'net_to_gross': net_to_gross,
            'thickness_ft': thickness
        }
    
    def generate_well_design(self, n_wells):
        """
        Generate well design parameters
        
        Parameters:
        -----------
        n_wells : int
            Number of wells to generate
            
        Returns:
        --------
        dict : Dictionary containing well design parameters
        """
        print("Generating well design parameters...")
        
        # Well depth (ft) - Normal distribution around typical depths
        well_depth = np.random.normal(8000, 1500, n_wells)
        well_depth = np.clip(well_depth, 4000, 15000)
        
        # Tubing diameter (inches) - Common sizes in industry
        tubing_sizes = [2.375, 2.875, 3.5, 4.5, 5.5]
        tubing_weights = [0.1, 0.3, 0.35, 0.2, 0.05]  # 3.5" most common
        tubing_diameter = np.random.choice(tubing_sizes, n_wells, p=tubing_weights)
        
        # Choke size (64ths of inch) - Production control
        choke_size = np.random.uniform(8, 64, n_wells)
        
        # Well type - Vertical, Horizontal, Deviated
        well_types = ['Vertical', 'Horizontal', 'Deviated']
        well_type_weights = [0.4, 0.5, 0.1]
        well_type = np.random.choice(well_types, n_wells, p=well_type_weights)
        
        # Completion type affects productivity
        completion_types = ['Perforated', 'Open Hole', 'Slotted Liner']
        completion_weights = [0.7, 0.2, 0.1]
        completion_type = np.random.choice(completion_types, n_wells, p=completion_weights)
        
        # Artificial lift type
        lift_types = ['Natural Flow', 'ESP', 'Rod Pump', 'Gas Lift']
        lift_weights = [0.3, 0.3, 0.25, 0.15]
        artificial_lift = np.random.choice(lift_types, n_wells, p=lift_weights)
        
        return {
            'well_depth_ft': well_depth,
            'tubing_diameter_in': tubing_diameter,
            'choke_size_64th': choke_size,
            'well_type': well_type,
            'completion_type': completion_type,
            'artificial_lift': artificial_lift
        }
    
    def generate_pressure_temperature(self, well_depth):
        """
        Generate pressure and temperature based on depth and regional gradients
        
        Parameters:
        -----------
        well_depth : array
            Well depths in feet
            
        Returns:
        --------
        dict : Dictionary containing pressure and temperature data
        """
        print("Generating pressure and temperature profiles...")
        
        n_wells = len(well_depth)
        
        # Pressure gradient: typical 0.43-0.47 psi/ft
        pressure_gradient = np.random.normal(0.45, 0.02, n_wells)
        reservoir_pressure = well_depth * pressure_gradient + np.random.normal(500, 100, n_wells)
        reservoir_pressure = np.clip(reservoir_pressure, 1500, 8000)
        
        # Temperature: surface temp + geothermal gradient
        surface_temp = np.random.normal(70, 10, n_wells)  # °F
        geothermal_gradient = np.random.normal(1.5, 0.2, n_wells)  # °F per 100 ft
        reservoir_temp = surface_temp + (well_depth / 100) * geothermal_gradient
        
        # Operating pressures
        bottomhole_pressure = reservoir_pressure - np.random.uniform(200, 1000, n_wells)
        bottomhole_pressure = np.clip(bottomhole_pressure, 200, reservoir_pressure * 0.9)
        
        wellhead_pressure = bottomhole_pressure - np.random.uniform(100, 600, n_wells)
        wellhead_pressure = np.clip(wellhead_pressure, 50, bottomhole_pressure * 0.8)
        
        return {
            'reservoir_pressure_psi': reservoir_pressure,
            'reservoir_temp_f': reservoir_temp,
            'bottomhole_pressure_psi': bottomhole_pressure,
            'wellhead_pressure_psi': wellhead_pressure
        }
    
    def generate_fluid_properties(self, n_wells, reservoir_temp):
        """
        Generate fluid properties based on reservoir conditions
        
        Parameters:
        -----------
        n_wells : int
            Number of wells
        reservoir_temp : array
            Reservoir temperatures
            
        Returns:
        --------
        dict : Dictionary containing fluid properties
        """
        print("Generating fluid properties...")
        
        # Oil gravity (API) - varies by region and formation
        oil_gravity = np.random.normal(35, 8, n_wells)
        oil_gravity = np.clip(oil_gravity, 15, 50)
        
        # Gas-Oil Ratio (scf/bbl) - correlated with reservoir pressure and temperature
        # Higher temperature/pressure generally means higher GOR
        base_gor = np.random.lognormal(5, 0.8, n_wells)
        temp_factor = (reservoir_temp - 150) / 100  # Temperature effect
        gor = base_gor * (1 + temp_factor * 0.3)
        gor = np.clip(gor, 50, 5000)
        
        # Water Cut (fraction) - represents well maturity
        # Uses beta distribution to create realistic water cut progression
        water_cut = np.random.beta(2, 8, n_wells) * 0.9
        
        # Formation Volume Factor - oil shrinkage from reservoir to surface
        fvf_oil = 1.1 + (gor / 10000) * 0.3  # Simplified correlation
        
        # Oil viscosity (cp) - function of API gravity and temperature
        oil_viscosity = 10 ** (3.0324 - 0.02023 * oil_gravity) * (reservoir_temp / 100) ** (-1.163)
        oil_viscosity = np.clip(oil_viscosity, 0.5, 100)
        
        return {
            'oil_gravity_api': oil_gravity,
            'gas_oil_ratio_scf_bbl': gor,
            'water_cut_fraction': water_cut,
            'fvf_oil': fvf_oil,
            'oil_viscosity_cp': oil_viscosity
        }
    
    def calculate_production_rates(self, reservoir_props, well_design, pressures, fluids):
        """
        Calculate production rates using petroleum engineering correlations
        
        Parameters:
        -----------
        reservoir_props : dict
            Reservoir properties
        well_design : dict
            Well design parameters
        pressures : dict
            Pressure data
        fluids : dict
            Fluid properties
            
        Returns:
        --------
        dict : Dictionary containing production rates
        """
        print("Calculating production rates using IPR correlations...")
        
        n_wells = len(reservoir_props['permeability_md'])
        
        # Productivity Index (bbl/day/psi) - Darcy's Law based
        # J = (0.00708 * k * h) / (μ * B * (ln(re/rw) + S))
        
        k = reservoir_props['permeability_md']
        h = reservoir_props['thickness_ft'] * reservoir_props['net_to_gross']
        mu = fluids['oil_viscosity_cp']
        B = fluids['fvf_oil']
        
        # Simplified productivity index calculation
        productivity_index = (0.007 * k * h) / (mu * B * 7)  # Assuming ln(re/rw)+S ≈ 7
        
        # Well type multiplier
        well_type_multiplier = np.ones(n_wells)
        horizontal_mask = well_design['well_type'] == 'Horizontal'
        well_type_multiplier[horizontal_mask] *= 3  # Horizontal wells more productive
        
        deviated_mask = well_design['well_type'] == 'Deviated'
        well_type_multiplier[deviated_mask] *= 1.5  # Deviated wells moderately more productive
        
        # Completion efficiency
        completion_efficiency = np.ones(n_wells)
        completion_efficiency[well_design['completion_type'] == 'Perforated'] *= 0.8
        completion_efficiency[well_design['completion_type'] == 'Open Hole'] *= 1.0
        completion_efficiency[well_design['completion_type'] == 'Slotted Liner'] *= 0.9
        
        # Pressure drawdown
        pressure_drawdown = pressures['reservoir_pressure_psi'] - pressures['bottomhole_pressure_psi']
        
        # Base oil rate using IPR
        oil_rate = (productivity_index * pressure_drawdown * well_type_multiplier * 
                   completion_efficiency * (1 - fluids['water_cut_fraction']))
        
        # Choke effect - limits maximum flow
        choke_factor = np.minimum(well_design['choke_size_64th'] / 32, 1.5)
        oil_rate *= choke_factor
        
        # Artificial lift effect
        lift_efficiency = np.ones(n_wells)
        lift_efficiency[well_design['artificial_lift'] == 'ESP'] *= 1.3
        lift_efficiency[well_design['artificial_lift'] == 'Rod Pump'] *= 1.1
        lift_efficiency[well_design['artificial_lift'] == 'Gas Lift'] *= 1.2
        oil_rate *= lift_efficiency
        
        # Add realistic noise and constraints
        oil_rate *= np.random.uniform(0.8, 1.2, n_wells)  # ±20% variability
        oil_rate = np.clip(oil_rate, 5, 2000)  # Realistic range
        
        # Gas production rate
        gas_rate = oil_rate * fluids['gas_oil_ratio_scf_bbl']
        
        # Water production rate
        water_rate = oil_rate * fluids['water_cut_fraction'] / (1 - fluids['water_cut_fraction'] + 1e-6)
        water_rate = np.clip(water_rate, 0, oil_rate * 10)
        
        return {
            'oil_rate_bbl_day': oil_rate,
            'gas_rate_scf_day': gas_rate,
            'water_rate_bbl_day': water_rate,
            'productivity_index': productivity_index
        }
    
    def generate_economic_data(self, n_wells, production_rates, well_design):
        """
        Generate economic parameters including prices and costs
        
        Parameters:
        -----------
        n_wells : int
            Number of wells
        production_rates : dict
            Production rate data
        well_design : dict
            Well design parameters
            
        Returns:
        --------
        dict : Dictionary containing economic data
        """
        print("Generating economic parameters...")
        
        # Commodity prices with regional variations
        oil_price_base = 75  # $/bbl
        oil_price_variation = np.random.normal(0, 10, n_wells)
        oil_price = oil_price_base + oil_price_variation
        oil_price = np.clip(oil_price, 45, 120)
        
        gas_price_base = 3.5  # $/Mcf
        gas_price_variation = np.random.normal(0, 0.8, n_wells)
        gas_price = gas_price_base + gas_price_variation
        gas_price = np.clip(gas_price, 1.5, 8.0)
        
        # Operating costs (OPEX)
        # Base daily cost varies by well type and artificial lift
        base_opex = np.full(n_wells, 200.0)  # Base cost $/day
        
        # Depth factor
        depth_factor = well_design['well_depth_ft'] * 0.01
        
        # Artificial lift cost
        lift_cost = np.zeros(n_wells)
        lift_cost[well_design['artificial_lift'] == 'ESP'] = 100
        lift_cost[well_design['artificial_lift'] == 'Rod Pump'] = 50
        lift_cost[well_design['artificial_lift'] == 'Gas Lift'] = 75
        
        # Variable costs based on production
        variable_cost_oil = production_rates['oil_rate_bbl_day'] * 2.5
        variable_cost_water = production_rates['water_rate_bbl_day'] * 1.0
        
        daily_opex = base_opex + depth_factor + lift_cost + variable_cost_oil + variable_cost_water
        
        # Capital costs (one-time, but affects economics)
        drilling_cost_per_ft = np.random.normal(150, 30, n_wells)
        drilling_cost = well_design['well_depth_ft'] * drilling_cost_per_ft
        
        completion_cost = np.random.normal(500000, 100000, n_wells)
        completion_cost = np.clip(completion_cost, 200000, 1500000)
        
        total_capex = drilling_cost + completion_cost
        
        # Revenue calculation
        oil_revenue = production_rates['oil_rate_bbl_day'] * oil_price
        gas_revenue = production_rates['gas_rate_scf_day'] * gas_price / 1000  # Convert to Mcf
        
        daily_revenue = oil_revenue + gas_revenue - daily_opex
        
        return {
            'oil_price_usd_bbl': oil_price,
            'gas_price_usd_mcf': gas_price,
            'daily_opex_usd': daily_opex,
            'drilling_cost_usd': drilling_cost,
            'completion_cost_usd': completion_cost,
            'total_capex_usd': total_capex,
            'daily_revenue_usd': daily_revenue
        }
    
    def calculate_performance_metrics(self, production_rates, economic_data, fluids):
        """
        Calculate key performance indicators
        
        Parameters:
        -----------
        production_rates : dict
            Production data
        economic_data : dict
            Economic data
        fluids : dict
            Fluid properties
            
        Returns:
        --------
        dict : Dictionary containing performance metrics
        """
        print("Calculating performance metrics...")
        
        oil_rate = production_rates['oil_rate_bbl_day']
        daily_revenue = economic_data['daily_revenue_usd']
        daily_opex = economic_data['daily_opex_usd']
        oil_price = economic_data['oil_price_usd_bbl']
        water_cut = fluids['water_cut_fraction']
        
        # Primary performance index (combines production efficiency and economics)
        performance_index = (oil_rate * (1 - water_cut) * oil_price / (daily_opex + 1)) * 100
        
        # Additional KPIs
        oil_cut = 1 - water_cut  # Fraction of oil in total liquid
        profit_per_barrel = daily_revenue / (oil_rate + 0.1)  # Avoid division by zero
        
        # Production efficiency (actual vs theoretical max)
        theoretical_max = oil_rate / (1 - water_cut + 0.001)  # If no water
        production_efficiency = oil_rate / theoretical_max
        
        # Economic efficiency
        economic_efficiency = np.where(daily_opex > 0, daily_revenue / daily_opex, 0)
        
        # Well ranking score (composite metric)
        ranking_score = (
            0.4 * (oil_rate / oil_rate.max()) +
            0.3 * (economic_efficiency / economic_efficiency.max()) +
            0.2 * (oil_cut / oil_cut.max()) +
            0.1 * (production_efficiency / production_efficiency.max())
        ) * 100
        
        return {
            'performance_index': performance_index,
            'oil_cut': oil_cut,
            'profit_per_barrel': profit_per_barrel,
            'production_efficiency': production_efficiency,
            'economic_efficiency': economic_efficiency,
            'ranking_score': ranking_score
        }
    
    def add_temporal_data(self, n_wells):
        """
        Add time-based information to wells
        
        Parameters:
        -----------
        n_wells : int
            Number of wells
            
        Returns:
        --------
        dict : Dictionary containing temporal data
        """
        print("Adding temporal data...")
        
        # Well vintage (when drilled)
        start_date = datetime(2015, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range = (end_date - start_date).days
        
        drill_dates = [start_date + timedelta(days=np.random.randint(0, date_range)) 
                      for _ in range(n_wells)]
        
        # Calculate well age in days
        current_date = datetime(2025, 1, 1)
        well_age_days = [(current_date - date).days for date in drill_dates]
        
        # Production months (time on production)
        production_months = np.random.uniform(1, 48, n_wells)  # 1-48 months
        
        # Days since last workover
        days_since_workover = np.random.exponential(180, n_wells)  # Average 6 months
        days_since_workover = np.clip(days_since_workover, 30, 1095)  # 1 month to 3 years
        
        return {
            'drill_date': drill_dates,
            'well_age_days': well_age_days,
            'production_months': production_months,
            'days_since_workover': days_since_workover
        }
    
    def generate_complete_dataset(self, n_wells=500, filename='well_data.csv'):
        """
        Generate complete well dataset with all parameters
        
        Parameters:
        -----------
        n_wells : int
            Number of wells to generate
        filename : str
            Output CSV filename
            
        Returns:
        --------
        pandas.DataFrame : Complete well dataset
        """
        print(f"\n{'='*50}")
        print(f"GENERATING WELL PERFORMANCE DATASET")
        print(f"{'='*50}")
        print(f"Number of wells: {n_wells}")
        print(f"Random seed: {self.random_seed}")
        print(f"Output file: {filename}")
        print()
        
        # Generate all data components
        reservoir_props = self.generate_reservoir_properties(n_wells)
        well_design = self.generate_well_design(n_wells)
        pressures = self.generate_pressure_temperature(well_design['well_depth_ft'])
        fluids = self.generate_fluid_properties(n_wells, pressures['reservoir_temp_f'])
        production = self.calculate_production_rates(reservoir_props, well_design, pressures, fluids)
        economics = self.generate_economic_data(n_wells, production, well_design)
        performance = self.calculate_performance_metrics(production, economics, fluids)
        temporal = self.add_temporal_data(n_wells)
        
        # Combine all data into DataFrame
        data_dict = {
            'well_id': range(1, n_wells + 1),
            **reservoir_props,
            **well_design,
            **pressures,
            **fluids,
            **production,
            **economics,
            **performance,
            **temporal
        }
        
        df = pd.DataFrame(data_dict)
        
        # Add some derived features
        df['pressure_drawdown'] = df['reservoir_pressure_psi'] - df['bottomhole_pressure_psi']
        df['total_liquid_rate'] = df['oil_rate_bbl_day'] + df['water_rate_bbl_day']
        df['productivity_factor'] = df['permeability_md'] * df['porosity_fraction']
        df['depth_category'] = pd.cut(df['well_depth_ft'], 
                                     bins=[0, 5000, 8000, 12000, float('inf')],
                                     labels=['Shallow', 'Medium', 'Deep', 'Ultra Deep'])
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        # Generate summary report
        self.generate_data_summary(df, filename)
        
        print(f"\n✅ Dataset successfully saved to {filename}")
        return df
    
    def generate_data_summary(self, df, filename):
        """
        Generate and display summary statistics for the dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Generated dataset
        filename : str
            Output filename
        """
        print(f"\n{'='*30}")
        print("DATASET SUMMARY")
        print(f"{'='*30}")
        
        print(f"Dataset shape: {df.shape}")
        print(f"File size: {os.path.getsize(filename) / 1024:.1f} KB")
        
        print(f"\nProduction Statistics:")
        print(f"  Average oil rate: {df['oil_rate_bbl_day'].mean():.1f} bbl/day")
        print(f"  Production range: {df['oil_rate_bbl_day'].min():.1f} - {df['oil_rate_bbl_day'].max():.1f} bbl/day")
        print(f"  Average water cut: {df['water_cut_fraction'].mean():.1%}")
        
        print(f"\nReservoir Statistics:")
        print(f"  Permeability range: {df['permeability_md'].min():.1f} - {df['permeability_md'].max():.1f} mD")
        print(f"  Average porosity: {df['porosity_fraction'].mean():.1%}")
        print(f"  Depth range: {df['well_depth_ft'].min():.0f} - {df['well_depth_ft'].max():.0f} ft")
        
        print(f"\nEconomic Statistics:")
        profitable_wells = len(df[df['daily_revenue_usd'] > 0])
        print(f"  Profitable wells: {profitable_wells} ({profitable_wells/len(df)*100:.1f}%)")
        print(f"  Average daily revenue: ${df['daily_revenue_usd'].mean():.0f}")
        print(f"  Average OPEX: ${df['daily_opex_usd'].mean():.0f}/day")
        
        print(f"\nWell Type Distribution:")
        well_type_counts = df['well_type'].value_counts()
        for well_type, count in well_type_counts.items():
            print(f"  {well_type}: {count} wells ({count/len(df)*100:.1f}%)")
        
        print(f"\nArtificial Lift Distribution:")
        lift_counts = df['artificial_lift'].value_counts()
        for lift_type, count in lift_counts.items():
            print(f"  {lift_type}: {count} wells ({count/len(df)*100:.1f}%)")
        
        print(f"\nPerformance Metrics:")
        print(f"  Average performance index: {df['performance_index'].mean():.1f}")
        print(f"  Performance range: {df['performance_index'].min():.1f} - {df['performance_index'].max():.1f}")
        print(f"  Average ranking score: {df['ranking_score'].mean():.1f}")

def create_visualization_report(filename='well_data.csv', output_file='data_summary.png'):
    """
    Create comprehensive visualization report of the generated data
    
    Parameters:
    -----------
    filename : str
        CSV filename to analyze
    output_file : str
        Output image filename
    """
    print("Creating visualization report...")
    
    # Load data
    df = pd.read_csv(filename)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(4, 3, figsize=(20, 24))
    fig.suptitle('Well Performance Dataset - Comprehensive Analysis', fontsize=20, fontweight='bold')
    
    # 1. Production rate distribution
    axes[0,0].hist(df['oil_rate_bbl_day'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0,0].set_title('Oil Production Rate Distribution', fontweight='bold')
    axes[0,0].set_xlabel('Oil Rate (bbl/day)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Reservoir quality
    scatter = axes[0,1].scatter(df['permeability_md'], df['porosity_fraction'], 
                               c=df['oil_rate_bbl_day'], alpha=0.6, s=30, cmap='viridis')
    axes[0,1].set_xlabel('Permeability (mD)')
    axes[0,1].set_ylabel('Porosity (fraction)')
    axes[0,1].set_title('Reservoir Quality vs Production', fontweight='bold')
    axes[0,1].set_xscale('log')
    plt.colorbar(scatter, ax=axes[0,1], label='Oil Rate (bbl/day)')
    
    # 3. Water cut impact
    axes[0,2].scatter(df['water_cut_fraction'], df['oil_rate_bbl_day'], alpha=0.5, color='red')
    axes[0,2].set_xlabel('Water Cut (fraction)')
    axes[0,2].set_ylabel('Oil Rate (bbl/day)')
    axes[0,2].set_title('Water Cut Impact on Production', fontweight='bold')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Economic performance
    colors = ['red' if x < 0 else 'green' for x in df['daily_revenue_usd']]
    axes[1,0].scatter(df['oil_rate_bbl_day'], df['daily_revenue_usd'], 
                     c=colors, alpha=0.6)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.7)
    axes[1,0].set_xlabel('Oil Rate (bbl/day)')
    axes[1,0].set_ylabel('Daily Revenue ($)')
    axes[1,0].set_title('Economic Performance', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Well type comparison
    well_type_data = [df[df['well_type'] == wt]['oil_rate_bbl_day'].values 
                      for wt in df['well_type'].unique()]
    well_type_labels = df['well_type'].unique()
    axes[1,1].boxplot(well_type_data, labels=well_type_labels)
    axes[1,1].set_title('Production by Well Type', fontweight='bold')
    axes[1,1].set_ylabel('Oil Rate (bbl/day)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # 6. Artificial lift comparison
    lift_type_data = [df[df['artificial_lift'] == lt]['oil_rate_bbl_day'].values 
                      for lt in df['artificial_lift'].unique()]
    lift_type_labels = df['artificial_lift'].unique()
    axes[1,2].boxplot(lift_type_data, labels=lift_type_labels)
    axes[1,2].set_title('Production by Artificial Lift', fontweight='bold')
    axes[1,2].set_ylabel('Oil Rate (bbl/day)')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    # 7. Depth vs production
    axes[2,0].scatter(df['well_depth_ft'], df['oil_rate_bbl_day'], alpha=0.5, color='blue')
    axes[2,0].set_xlabel('Well Depth (ft)')
    axes[2,0].set_ylabel('Oil Rate (bbl/day)')
    axes[2,0].set_title('Depth vs Production', fontweight='bold')
    axes[2,0].grid(True, alpha=0.3)
    
    # 8. Performance index distribution
    axes[2,1].hist(df['performance_index'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[2,1].set_title('Performance Index Distribution', fontweight='bold')
    axes[2,1].set_xlabel('Performance Index')
    axes[2,1].set_ylabel('Frequency')
    axes[2,1].grid(True, alpha=0.3)
    
    # 9. Correlation heatmap
    corr_cols = ['permeability_md', 'porosity_fraction', 'water_cut_fraction', 
                 'oil_rate_bbl_day', 'daily_revenue_usd', 'performance_index']
    corr_matrix = df[corr_cols].corr()
    im = axes[2,2].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[2,2].set_xticks(range(len(corr_cols)))
    axes[2,2].set_yticks(range(len(corr_cols)))
    axes[2,2].set_xticklabels([col.replace('_', '\n') for col in corr_cols], rotation=45, ha='right')
    axes[2,2].set_yticklabels([col.replace('_', '\n') for col in corr_cols])
    axes[2,2].set_title('Feature Correlation Matrix', fontweight='bold')
    
    # Add correlation values to heatmap
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            axes[2,2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    # 10. Pressure analysis
    axes[3,0].scatter(df['pressure_drawdown'], df['oil_rate_bbl_day'], alpha=0.5, color='purple')
    axes[3,0].set_xlabel('Pressure Drawdown (psi)')
    axes[3,0].set_ylabel('Oil Rate (bbl/day)')
    axes[3,0].set_title('Pressure Drawdown vs Production', fontweight='bold')
    axes[3,0].grid(True, alpha=0.3)
    
    # 11. Economic efficiency
    axes[3,1].scatter(df['daily_opex_usd'], df['daily_revenue_usd'], alpha=0.5, color='brown')
    axes[3,1].plot([df['daily_opex_usd'].min(), df['daily_opex_usd'].max()], 
                   [df['daily_opex_usd'].min(), df['daily_opex_usd'].max()], 
                   'r--', label='Break-even line')
    axes[3,1].set_xlabel('Daily OPEX ($)')
    axes[3,1].set_ylabel('Daily Revenue ($)')
    axes[3,1].set_title('Revenue vs Operating Costs', fontweight='bold')
    axes[3,1].legend()
    axes[3,1].grid(True, alpha=0.3)
    
    # 12. Well age analysis
    axes[3,2].scatter(df['well_age_days'], df['oil_rate_bbl_day'], alpha=0.5, color='gray')
    axes[3,2].set_xlabel('Well Age (days)')
    axes[3,2].set_ylabel('Oil Rate (bbl/day)')
    axes[3,2].set_title('Well Age vs Production', fontweight='bold')
    axes[3,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization report saved to {output_file}")
    plt.show()

def validate_dataset(filename='well_data.csv'):
    """
    Validate the generated dataset for consistency and realistic values
    
    Parameters:
    -----------
    filename : str
        CSV filename to validate
        
    Returns:
    --------
    dict : Validation results
    """
    print("\nValidating dataset...")
    
    df = pd.read_csv(filename)
    validation_results = {}
    
    # Check for missing values
    missing_values = df.isnull().sum()
    validation_results['missing_values'] = missing_values[missing_values > 0].to_dict()
    
    # Check for realistic ranges
    range_checks = {
        'permeability_md': (0.1, 10000),
        'porosity_fraction': (0.01, 0.4),
        'oil_rate_bbl_day': (0, 5000),
        'water_cut_fraction': (0, 1),
        'reservoir_pressure_psi': (500, 10000),
        'well_depth_ft': (1000, 20000),
        'oil_gravity_api': (10, 60)
    }
    
    range_violations = {}
    for column, (min_val, max_val) in range_checks.items():
        if column in df.columns:
            violations = df[(df[column] < min_val) | (df[column] > max_val)]
            if len(violations) > 0:
                range_violations[column] = len(violations)
    
    validation_results['range_violations'] = range_violations
    
    # Check for physical consistency
    consistency_checks = []
    
    # Pressure consistency: reservoir > bottomhole > wellhead
    pressure_inconsistent = df[
        (df['reservoir_pressure_psi'] <= df['bottomhole_pressure_psi']) |
        (df['bottomhole_pressure_psi'] <= df['wellhead_pressure_psi'])
    ]
    if len(pressure_inconsistent) > 0:
        consistency_checks.append(f"Pressure hierarchy violated in {len(pressure_inconsistent)} wells")
    
    # Production consistency: oil rate should be positive for producing wells
    zero_production = df[df['oil_rate_bbl_day'] <= 0]
    if len(zero_production) > 0:
        consistency_checks.append(f"{len(zero_production)} wells with zero/negative oil production")
    
    # Economic consistency: very high losses should be rare
    high_losses = df[df['daily_revenue_usd'] < -1000]
    if len(high_losses) > len(df) * 0.05:  # More than 5% with high losses
        consistency_checks.append(f"{len(high_losses)} wells with very high daily losses (>${1000})")
    
    validation_results['consistency_issues'] = consistency_checks
    
    # Statistical validation
    stats_validation = {}
    
    # Check if distributions look reasonable
    if df['oil_rate_bbl_day'].std() / df['oil_rate_bbl_day'].mean() < 0.2:
        stats_validation['low_production_variability'] = "Production rates may be too uniform"
    
    if df['permeability_md'].min() / df['permeability_md'].max() > 0.1:
        stats_validation['narrow_permeability_range'] = "Permeability range may be too narrow"
    
    validation_results['statistical_issues'] = stats_validation
    
    # Print validation summary
    print(f"\n{'='*30}")
    print("VALIDATION SUMMARY")
    print(f"{'='*30}")
    
    if not validation_results['missing_values']:
        print("✅ No missing values found")
    else:
        print("❌ Missing values detected:")
        for col, count in validation_results['missing_values'].items():
            print(f"   {col}: {count} missing")
    
    if not validation_results['range_violations']:
        print("✅ All values within realistic ranges")
    else:
        print("⚠️  Range violations detected:")
        for col, count in validation_results['range_violations'].items():
            print(f"   {col}: {count} violations")
    
    if not validation_results['consistency_issues']:
        print("✅ Physical consistency checks passed")
    else:
        print("⚠️  Consistency issues detected:")
        for issue in validation_results['consistency_issues']:
            print(f"   {issue}")
    
    if not validation_results['statistical_issues']:
        print("✅ Statistical distributions look reasonable")
    else:
        print("⚠️  Statistical issues detected:")
        for issue, description in validation_results['statistical_issues'].items():
            print(f"   {description}")
    
    return validation_results

def create_custom_dataset(config, filename='custom_well_data.csv'):
    """
    Create a custom dataset based on user configuration
    
    Parameters:
    -----------
    config : dict
        Configuration parameters for dataset generation
    filename : str
        Output filename
        
    Returns:
    --------
    pandas.DataFrame : Custom dataset
    """
    print("Creating custom dataset based on configuration...")
    
    # Default configuration
    default_config = {
        'n_wells': 500,
        'reservoir_type': 'conventional',  # 'conventional', 'tight', 'heavy_oil'
        'field_maturity': 'mature',        # 'new', 'developing', 'mature', 'declining'
        'geographic_region': 'north_america', # 'north_america', 'middle_east', 'north_sea'
        'water_cut_bias': 'medium',        # 'low', 'medium', 'high'
        'economic_environment': 'normal',   # 'low_price', 'normal', 'high_price'
        'random_seed': 42
    }
    
    # Merge with user config
    config = {**default_config, **config}
    
    # Initialize generator with custom seed
    generator = WellDataGenerator(random_seed=config['random_seed'])
    
    # Modify generation parameters based on configuration
    
    # Reservoir type adjustments
    if config['reservoir_type'] == 'tight':
        # Tight reservoirs: lower permeability, lower production
        print("Configuring for tight reservoir characteristics...")
    elif config['reservoir_type'] == 'heavy_oil':
        # Heavy oil: higher viscosity, different production profiles
        print("Configuring for heavy oil reservoir characteristics...")
    
    # Field maturity adjustments
    if config['field_maturity'] == 'declining':
        # Older fields: higher water cuts, lower pressures
        print("Configuring for declining field characteristics...")
    elif config['field_maturity'] == 'new':
        # New fields: lower water cuts, higher pressures
        print("Configuring for new field characteristics...")
    
    # Generate the dataset with custom parameters
    df = generator.generate_complete_dataset(
        n_wells=config['n_wells'], 
        filename=filename
    )
    
    # Apply post-processing adjustments based on config
    df = apply_custom_adjustments(df, config)
    
    # Save modified dataset
    df.to_csv(filename, index=False)
    print(f"Custom dataset saved to {filename}")
    
    return df

def apply_custom_adjustments(df, config):
    """
    Apply custom adjustments to dataset based on configuration
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Generated dataset
    config : dict
        Configuration parameters
        
    Returns:
    --------
    pandas.DataFrame : Adjusted dataset
    """
    print("Applying custom adjustments...")
    
    # Reservoir type adjustments
    if config['reservoir_type'] == 'tight':
        df['permeability_md'] *= 0.1  # Much lower permeability
        df['oil_rate_bbl_day'] *= 0.3  # Lower production rates
        df['gas_oil_ratio_scf_bbl'] *= 2  # Higher GOR typical for tight oil
    
    elif config['reservoir_type'] == 'heavy_oil':
        df['oil_gravity_api'] = np.clip(df['oil_gravity_api'] * 0.6, 8, 20)  # Heavier oil
        df['oil_viscosity_cp'] *= 10  # Much higher viscosity
        df['oil_rate_bbl_day'] *= 0.5  # Lower production rates
    
    # Field maturity adjustments
    if config['field_maturity'] == 'declining':
        df['water_cut_fraction'] = np.clip(df['water_cut_fraction'] + 0.3, 0, 0.95)
        df['reservoir_pressure_psi'] *= 0.7  # Pressure depletion
        df['oil_rate_bbl_day'] *= 0.6  # Production decline
    
    elif config['field_maturity'] == 'new':
        df['water_cut_fraction'] = np.clip(df['water_cut_fraction'] * 0.3, 0, 0.4)
        df['reservoir_pressure_psi'] *= 1.2  # Higher initial pressure
        df['oil_rate_bbl_day'] *= 1.3  # Higher initial production
    
    # Water cut bias adjustments
    if config['water_cut_bias'] == 'high':
        df['water_cut_fraction'] = np.clip(df['water_cut_fraction'] + 0.2, 0, 0.95)
    elif config['water_cut_bias'] == 'low':
        df['water_cut_fraction'] = np.clip(df['water_cut_fraction'] * 0.5, 0, 0.6)
    
    # Economic environment adjustments
    if config['economic_environment'] == 'low_price':
        df['oil_price_usd_bbl'] *= 0.6  # Lower oil prices
        df['gas_price_usd_mcf'] *= 0.7  # Lower gas prices
    elif config['economic_environment'] == 'high_price':
        df['oil_price_usd_bbl'] *= 1.4  # Higher oil prices
        df['gas_price_usd_mcf'] *= 1.3  # Higher gas prices
    
    # Recalculate dependent variables
    df['water_rate_bbl_day'] = df['oil_rate_bbl_day'] * df['water_cut_fraction'] / (1 - df['water_cut_fraction'] + 1e-6)
    df['gas_rate_scf_day'] = df['oil_rate_bbl_day'] * df['gas_oil_ratio_scf_bbl']
    
    # Recalculate economics
    oil_revenue = df['oil_rate_bbl_day'] * df['oil_price_usd_bbl']
    gas_revenue = df['gas_rate_scf_day'] * df['gas_price_usd_mcf'] / 1000
    df['daily_revenue_usd'] = oil_revenue + gas_revenue - df['daily_opex_usd']
    
    # Recalculate performance metrics
    df['performance_index'] = (df['oil_rate_bbl_day'] * (1 - df['water_cut_fraction']) * 
                              df['oil_price_usd_bbl'] / (df['daily_opex_usd'] + 1)) * 100
    
    return df

# Import required libraries for file operations
import os

# Example usage functions
def main():
    """
    Main function demonstrating data preparation capabilities
    """
    print("Well Performance Data Preparation Module")
    print("=" * 50)
    
    # Example 1: Generate standard dataset
    print("\n1. Generating standard dataset...")
    generator = WellDataGenerator(random_seed=42)
    df_standard = generator.generate_complete_dataset(
        n_wells=500, 
        filename='well_data_standard.csv'
    )
    
    # Example 2: Generate custom datasets for different scenarios
    print("\n2. Generating custom datasets...")
    
    # Tight oil scenario
    tight_config = {
        'n_wells': 200,
        'reservoir_type': 'tight',
        'field_maturity': 'new',
        'water_cut_bias': 'low',
        'economic_environment': 'normal'
    }
    df_tight = create_custom_dataset(tight_config, 'well_data_tight_oil.csv')
    
    # Mature field scenario
    mature_config = {
        'n_wells': 300,
        'reservoir_type': 'conventional',
        'field_maturity': 'declining',
        'water_cut_bias': 'high',
        'economic_environment': 'low_price'
    }
    df_mature = create_custom_dataset(mature_config, 'well_data_mature_field.csv')
    
    # Example 3: Validate datasets
    print("\n3. Validating generated datasets...")
    validate_dataset('well_data_standard.csv')
    
    # Example 4: Create visualization reports
    print("\n4. Creating visualization reports...")
    create_visualization_report('well_data_standard.csv', 'standard_dataset_analysis.png')
    
    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print("Generated files:")
    print("  - well_data_standard.csv (500 wells)")
    print("  - well_data_tight_oil.csv (200 wells)")
    print("  - well_data_mature_field.csv (300 wells)")
    print("  - standard_dataset_analysis.png (visualization report)")

if __name__ == "__main__":
    main()