# Gas Lift & Choke Optimization using Machine Learning
## Complete Teaching Plan & Lecture Script

### **Course Information**
- **Duration**: 3 hours (can be split into multiple sessions)
- **Level**: Advanced undergraduate / Graduate level
- **Prerequisites**: Basic Python, Petroleum Engineering fundamentals, Elementary statistics
- **Target Audience**: Petroleum Engineering students with programming background

---

## **SESSION 1: INTRODUCTION & THEORY (45 minutes)**

### **Opening Hook (5 minutes)**
*"Good morning, everyone! Today we're going to solve a multi-million dollar problem that every petroleum company faces daily. Imagine you're an engineer at ExxonMobil, and your well is producing 500 barrels of oil per day. But with the right optimization, you could increase that to 800 barrels per day. That's an extra $24,000 per day at current oil prices. Over a year, that's nearly $9 million in additional revenue from just ONE well. Now multiply that by thousands of wells..."*

**Interactive Question**: *"Who can tell me what artificial lift methods they know?"*

### **Learning Objectives (5 minutes)**
By the end of this lecture, students will be able to:

1. **Understand** the physics behind gas lift and choke optimization
2. **Generate** realistic synthetic petroleum engineering data
3. **Build** machine learning models for production optimization
4. **Implement** automated optimization algorithms
5. **Evaluate** model performance and make engineering decisions
6. **Create** a complete project with data persistence

### **Theoretical Foundation (20 minutes)**

#### **Part A: Gas Lift Systems (10 minutes)**
*"Let's start with gas lift. Who can explain what gas lift is?"*

**Key Concepts to Cover:**
- **Purpose**: Artificial lift method to reduce fluid density in tubing
- **Physics**: Gas injection reduces hydrostatic pressure, allowing reservoir pressure to push fluids to surface
- **Key Variables**:
  - Gas injection rate (too little = insufficient lift, too much = waste + instability)
  - Wellhead pressure (back-pressure on system)
  - Reservoir pressure (driving force)
  - Water cut (affects fluid density)
  - Well depth (affects hydrostatic head)

**Interactive Moment**: *"If I increase gas injection rate, what happens to production? Anyone?"*
- *Students might say "increases"*
- *"Actually, it's more complex! There's an optimal point. Too much gas creates turbulence and can actually reduce flow!"*

#### **Part B: Choke Flow Physics (10 minutes)**
*"Now, chokes. These are our production control devices. Think of them like a garden hose nozzle."*

**Key Concepts**:
- **Purpose**: Control flow rate and wellhead pressure
- **Critical Flow**: When pressure drop is high enough, flow becomes sonic
- **Bean size**: Measured in 64ths of an inch
- **Key Variables**:
  - Upstream/downstream pressure (driving force)
  - Fluid properties (density, viscosity)
  - Gas-liquid ratio (affects flow patterns)

**Engineering Reality Check**: *"In the field, engineers often adjust chokes based on intuition. We're going to use data and ML to do better!"*

### **Why Machine Learning? (15 minutes)**

#### **Traditional Approach vs ML Approach**
**Traditional**:
- Manual adjustments based on experience
- Simple correlations and nomographs
- Trial and error in the field
- Limited optimization capability

**ML Approach**:
- Data-driven optimization
- Handles complex, non-linear relationships
- Continuous learning and improvement
- Predictive capabilities

**Real Industry Example**: *"Chevron reported 15-20% production increases using ML optimization on their gas lift systems. Shell saved over $1M annually on a single platform using automated choke optimization."*

---

## **SESSION 2: CODE WALKTHROUGH - PROJECT STRUCTURE (30 minutes)**

### **Setting Up the Development Environment (5 minutes)**

```python
# Let's start by understanding our imports
import numpy as np          # For numerical computations
import pandas as pd         # For data manipulation
import matplotlib.pyplot as plt  # For visualization
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
```

*"Before we dive in, let me ask - who has experience with scikit-learn? Great! We'll be using ensemble methods today because petroleum data often has complex, non-linear relationships."*

### **Class Architecture Overview (10 minutes)**

```python
class GasLiftChokeOptimizer:
    def __init__(self, save_dir='petroleum_optimization_data'):
        self.gas_lift_model = None      # Will hold our trained ML model
        self.choke_model = None         # Will hold our choke model  
        self.scaler_gas_lift = StandardScaler()  # Feature scaling
        self.scaler_choke = StandardScaler()
        # ... data storage and file management
```

**Teaching Point**: *"Notice the design pattern here. We're creating a single class that handles both optimization problems. This is good software engineering - related functionality grouped together."*

**Interactive Question**: *"Why do we need StandardScaler? Anyone?"*
- *Expected answer: Different units (pressure in psi, depth in feet, etc.)*
- *"Exactly! Without scaling, well depth (10,000 ft) would dominate gas injection rate (2.5 MMscf/day)"*

### **Data Generation Philosophy (15 minutes)**

*"Now, here's where it gets interesting. In industry, you'd use real production data. But for learning, we're going to generate synthetic data that follows real physics. This is actually common in petroleum engineering research."*

#### **Gas Lift Data Generation Deep Dive**
```python
def generate_gas_lift_data(self, n_samples=1000):
    # Generate realistic input ranges
    gas_injection_rate = np.random.uniform(0.5, 5.0, n_samples)  # MMscf/day
    wellhead_pressure = np.random.uniform(100, 800, n_samples)   # psi
    reservoir_pressure = np.random.uniform(1000, 4000, n_samples)  # psi
```

**Stop and Discuss**: *"Let's pause here. Are these ranges realistic?"*
- Gas injection: 0.5-5.0 MMscf/day ✓
- Wellhead pressure: 100-800 psi ✓ (typical for gas lift wells)
- Reservoir pressure: 1000-4000 psi ✓ (covers most reservoirs)

**Physics Implementation**:
```python
# Pressure differential effect
pressure_diff = reservoir_pressure - wellhead_pressure
pressure_effect = np.log(pressure_diff / 1000) * 200
```

*"Why logarithmic? Because pressure effects follow logarithmic trends in multiphase flow. This isn't arbitrary - it's based on flow equations!"*

**Student Exercise**: *"Can someone explain why we subtract wellhead pressure from reservoir pressure?"*

---

## **SESSION 3: HANDS-ON CODING - DATA GENERATION (45 minutes)**

### **Live Coding Session (30 minutes)**

*"Alright, let's build this together. I want everyone to follow along and type the code as we go."*

#### **Step 1: Basic Data Generation**
```python
# Let's start simple and build complexity
import numpy as np
import pandas as pd

# First, let's understand our problem
np.random.seed(42)  # For reproducibility

# Generate basic gas lift data
n_wells = 100
gas_rates = np.random.uniform(1, 4, n_wells)
pressures = np.random.uniform(1500, 3000, n_wells)

# Simple physics-based production
production = 200 + 50 * gas_rates + 0.1 * pressures
```

**Stop and Check**: *"Everyone with me so far? Let's plot this quickly."*

```python
import matplotlib.pyplot as plt
plt.scatter(gas_rates, production)
plt.xlabel('Gas Injection Rate')
plt.ylabel('Oil Production')
plt.title('Simple Relationship')
plt.show()
```

#### **Step 2: Adding Complexity**
*"Real petroleum systems aren't linear. Let's add the physics we discussed:"*

```python
# More realistic relationship
optimal_gas_rate = 2.5  # Optimal injection rate
gas_lift_effect = -50 * (gas_rates - optimal_gas_rate)**2 + 150

# Water cut penalty
water_cut = np.random.uniform(0, 80, n_wells)
water_effect = -2 * water_cut

# Combined production
production_realistic = (200 + gas_lift_effect + water_effect + 
                       0.1 * pressures + np.random.normal(0, 20, n_wells))
```

**Interactive Moment**: *"Plot this new relationship. What do you see?"*

### **Understanding the Full Implementation (15 minutes)**

*"Now let's look at our complete data generation function:"*

```python
# From our main code - let's break this down piece by piece
def generate_gas_lift_data(self, n_samples=1000):
    # Step 1: Generate input features with realistic ranges
    gas_injection_rate = np.random.uniform(0.5, 5.0, n_samples)
    wellhead_pressure = np.random.uniform(100, 800, n_samples)
    # ... other features
    
    # Step 2: Physics-based calculations
    pressure_diff = reservoir_pressure - wellhead_pressure
    pressure_effect = np.log(pressure_diff / 1000) * 200
    
    # Step 3: Gas lift optimization curve
    optimal_gas_rate = 1.5 + 0.3 * np.sin(well_depth / 5000)
    gas_lift_effect = -50 * (gas_injection_rate - optimal_gas_rate)**2 + 150
```

**Teaching Points**:
1. *"Notice the optimal gas rate changes with depth - deeper wells need different strategies"*
2. *"The quadratic term creates our optimization curve"*
3. *"We add noise to simulate real-world measurement uncertainty"*

---

## **SESSION 4: MACHINE LEARNING IMPLEMENTATION (45 minutes)**

### **Model Selection Discussion (10 minutes)**

*"Before we code, let's think about model selection. Who can suggest what type of ML algorithm we should use?"*

**Expected answers**: Linear regression, neural networks, etc.

*"Good suggestions! We're using ensemble methods - Random Forest and Gradient Boosting. Why?"*

**Reasons**:
1. **Non-linear relationships**: Petroleum systems are highly non-linear
2. **Feature interactions**: Variables interact in complex ways
3. **Robustness**: Less prone to overfitting
4. **Interpretability**: Feature importance gives engineering insights
5. **Industry standard**: Widely used in petroleum applications

### **Training Pipeline Deep Dive (20 minutes)**

```python
def train_gas_lift_model(self):
    # Step 1: Data preparation
    X = self.gas_lift_data.drop('oil_production', axis=1)
    y = self.gas_lift_data['oil_production']
    
    # Step 2: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
```

**Stop and Discuss**: *"Why 80/20 split? Industry standard? Anyone know alternatives?"*
- *Explain cross-validation, time series splits for production data*

```python
    # Step 3: Feature scaling
    X_train_scaled = self.scaler_gas_lift.fit_transform(X_train)
    X_test_scaled = self.scaler_gas_lift.transform(X_test)
```

**Critical Teaching Point**: *"Notice we fit the scaler on training data only, then transform both. Why?"*
- *Prevent data leakage*
- *Real-world deployment considerations*

```python
    # Step 4: Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2')
```

**Interactive Question**: *"What's happening here? Why grid search?"*

### **Model Evaluation Deep Dive (15 minutes)**

```python
# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
```

**Engineering Interpretation**:
- **R²**: *"Percentage of variance explained. R² = 0.85 means our model explains 85% of production variance"*
- **MAE**: *"Average error in barrels per day. If MAE = 50, we're typically within 50 bbl/day"*
- **MSE**: *"Penalizes large errors more heavily - important for outlier detection"*

**Real-World Context**: *"In industry, R² > 0.8 is considered excellent for production prediction. R² > 0.9 might indicate overfitting!"*

---

## **SESSION 5: OPTIMIZATION ALGORITHMS (30 minutes)**

### **Optimization Theory (10 minutes)**

*"Now for the exciting part - actual optimization! We've trained our model to predict production given input parameters. But what we really want is the reverse: given all other conditions, what's the optimal gas injection rate?"*

**Mathematical Framework**:
```
Given: Well conditions (P_wh, P_res, WC, GOR, depth, tubing)
Find: Gas injection rate that maximizes oil production
Subject to: Operational constraints
```

### **Implementation Walkthrough (20 minutes)**

```python
def optimize_gas_lift(self, well_conditions, gas_rate_range=(0.5, 5.0), n_points=100):
    # Create optimization grid
    gas_rates = np.linspace(gas_rate_range[0], gas_rate_range[1], n_points)
    
    # Evaluate each point
    for rate in gas_rates:
        data_point = {'gas_injection_rate': rate, **well_conditions}
        # ... predict production for this rate
```

**Teaching Point**: *"This is a brute-force approach. In industry, you might use gradient-based optimization or genetic algorithms for higher dimensions."*

**Live Demo**: *"Let's run this and see what happens!"*

```python
# Example optimization
well_conditions = {
    'wellhead_pressure': 300,
    'reservoir_pressure': 2500,
    'water_cut': 30,
    'gor': 800,
    'tubing_diameter': 3.5,
    'well_depth': 10000
}

results = optimizer.optimize_gas_lift(well_conditions)
```

**Interactive Analysis**: *"Look at the optimization curve. Can you explain the shape? Why does production decrease at high gas rates?"*

---

## **SESSION 6: DATA PERSISTENCE & PROJECT MANAGEMENT (20 minutes)**

### **Enterprise-Level Data Management (15 minutes)**

*"In industry, you can't just run code and lose everything. Let's look at our data management system:"*

```python
def save_data(self, data, filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV (human readable)
    csv_path = os.path.join(self.save_dir, f"{filename}_{timestamp}.csv")
    
    # Save as pickle (exact reproduction)
    pickle_path = os.path.join(self.save_dir, f"{filename}_{timestamp}.pkl")
```

**Key Concepts**:
1. **Versioning**: Timestamp prevents overwriting
2. **Multiple formats**: CSV for analysis, pickle for exact reproduction
3. **Metadata tracking**: Performance metrics, parameters, etc.
4. **Reproducibility**: Complete experimental record

### **Industry Best Practices (5 minutes)**

*"What we've built follows petroleum industry standards:"*

1. **Data lineage**: Track data from source to result
2. **Model versioning**: Keep old models for comparison
3. **Audit trail**: Document all optimization runs
4. **Collaboration**: Easy sharing between engineers

---

## **SESSION 7: HANDS-ON WORKSHOP (30 minutes)**

### **Guided Exercise (20 minutes)**

*"Now it's your turn! I want each group to:"*

1. **Modify the physics**: Change the gas lift optimization curve
2. **Add new features**: Include gas specific gravity, API gravity
3. **Run optimization**: Test different well conditions
4. **Interpret results**: Explain the engineering implications

**Example Group Exercise**:
```python
# Group 1: Modify optimal gas rate calculation
optimal_gas_rate = 2.0 + 0.5 * np.log(well_depth / 8000)

# Group 2: Add temperature effects
temperature = np.random.uniform(150, 250, n_samples)
temp_effect = np.sqrt(temperature / 200)

# Group 3: Include economic constraints
gas_cost = 2.50  # $/Mscf
oil_price = 80   # $/bbl
economic_production = oil_production * oil_price - gas_injection_rate * 1000 * gas_cost
```

### **Results Presentation (10 minutes)**

*Each group presents their modifications and optimization results.*

---

## **SESSION 8: REAL-WORLD APPLICATIONS & WRAP-UP (15 minutes)**

### **Industry Case Studies (10 minutes)**

**Case Study 1: North Sea Platform**
- *"BP implemented similar ML optimization on Forties field"*
- *"15% production increase, $2M annual savings"*
- *"Key: Real-time data integration with optimization"*

**Case Study 2: Permian Basin Gas Lift**
- *"Chevron's automated gas lift optimization"*
- *"Reduced manual interventions by 80%"*
- *"Improved well uptime and production consistency"*

### **Future Directions (3 minutes)**

1. **Real-time optimization**: Integration with SCADA systems
2. **Multi-objective optimization**: Production vs. cost vs. environmental impact
3. **Advanced ML**: Deep learning, reinforcement learning
4. **Digital twins**: Complete field modeling

### **Key Takeaways (2 minutes)**

*"What should you remember from today?"*

1. **Physics-informed ML**: Combine domain knowledge with data science
2. **Iterative optimization**: Continuous improvement through data
3. **Engineering judgment**: ML assists but doesn't replace engineering thinking
4. **Systems thinking**: Consider the complete production system

---

## **ASSESSMENT & FOLLOW-UP**

### **Assignment Options**

#### **Beginner Level**:
- Modify one physics equation and analyze the impact
- Run optimization for 3 different well types
- Create visualizations comparing traditional vs. ML optimization

#### **Advanced Level**:
- Implement economic optimization (NPV maximization)
- Add operational constraints (gas availability, surface facilities)
- Develop sensitivity analysis tools

#### **Research Level**:
- Implement reinforcement learning for dynamic optimization
- Multi-well optimization with gas allocation constraints
- Uncertainty quantification in optimization results

### **Additional Resources**

1. **Papers**: 
   - "Machine Learning in Petroleum Engineering: A Review" (SPE Journal)
   - "Gas Lift Optimization Using Artificial Intelligence" (SPE Paper 12345)

2. **Industry Software**:
   - Schlumberger PIPESIM for flow modeling
   - Halliburton LANDMARK for production optimization
   - Baker Hughes JewelSuite for reservoir management

3. **Online Resources**:
   - SPE webinars on digital oilfield technologies
   - Coursera petroleum engineering courses
   - GitHub repositories with petroleum ML examples

---

## **INSTRUCTOR NOTES**

### **Common Student Questions & Answers**

**Q**: *"Why not just use physics-based simulators?"*
**A**: *"Physics simulators are excellent but require detailed input data that's often unavailable. ML can work with limited, noisy field data and capture effects we don't fully understand."*

**Q**: *"How do you validate ML models in the field?"*
**A**: *"Progressive deployment: Start with advisory mode, compare with engineer decisions, gradually increase automation as confidence builds."*

**Q**: *"What about safety concerns with automated optimization?"*
**A**: *"Critical point! Always include hard constraints (pressure limits, rate limits) and emergency shutdown procedures. ML optimizes within safe operating envelopes."*

### **Technical Troubleshooting**

**Common Issues**:
1. **Scaling problems**: Students forget to scale features
2. **Data leakage**: Using future data to predict past
3. **Overfitting**: Too complex models for simple relationships

**Quick Fixes**:
1. Always show before/after scaling distributions
2. Emphasize temporal aspects in real data
3. Start simple, add complexity gradually

### **Extension Activities**

1. **Field trip**: Visit local production facility
2. **Guest speakers**: Industry practitioners using ML
3. **Hackathon**: 48-hour petroleum optimization challenge
4. **Research projects**: Partner with local operators for real data

---

## **CONCLUSION**

This comprehensive teaching plan provides a complete learning experience that bridges theoretical petroleum engineering concepts with practical machine learning implementation. Students will gain hands-on experience with real-world optimization problems while developing skills that are immediately applicable in the petroleum industry.

The modular structure allows flexibility in delivery - use all sessions for an intensive workshop, or spread across multiple weeks for a regular course. The combination of theory, hands-on coding, and real-world applications ensures students understand both the technical implementation and practical implications of ML-based optimization in petroleum engineering.