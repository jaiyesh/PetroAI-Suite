# Teaching Plan: Well Spacing Optimization for Infill Wells Using Machine Learning

## Course Information
- **Duration**: 2-3 hours (can be split into multiple sessions)
- **Target Audience**: Petroleum engineering students, data science students, professionals
- **Prerequisites**: Basic Python knowledge, introductory statistics, petroleum engineering fundamentals
- **Learning Objectives**: 
  - Understand well spacing optimization challenges
  - Apply machine learning to petroleum engineering problems
  - Interpret and visualize reservoir data
  - Make data-driven drilling decisions

---

## Teaching Script & Presentation Structure

### **SECTION 1: INTRODUCTION (15 minutes)**

#### **Opening Hook**
*"Imagine you're a petroleum engineer at an oil company. You have an existing field with producing wells, and management wants to drill additional 'infill' wells to increase production. The question is: Where should you place these new wells to maximize oil recovery while minimizing interference with existing wells? This is a multi-million dollar decision that we'll solve using machine learning today."*

#### **Learning Objectives Presentation**
> **SAY TO STUDENTS:**
> 
> "By the end of this session, you will be able to:
> 1. Understand the petroleum engineering principles behind well spacing
> 2. Generate synthetic reservoir data using realistic parameters
> 3. Apply multiple machine learning algorithms to optimize well placement
> 4. Interpret results and make business recommendations
> 5. Create professional visualizations for technical presentations"

#### **Real-World Context**
> **EXPLAIN:**
> 
> "Well spacing optimization is critical because:
> - **Economic Impact**: Each well costs $2-5 million to drill and complete
> - **Technical Challenge**: Wells too close together interfere with each other
> - **Regulatory Requirements**: Spacing rules vary by location
> - **Data Complexity**: Multiple geological, engineering, and economic factors
> - **Risk Management**: Wrong decisions can cost millions in lost production"

---

### **SECTION 2: PETROLEUM ENGINEERING FUNDAMENTALS (20 minutes)**

#### **Key Concepts Review**
> **INTERACTIVE QUESTION:** *"Who can tell me what factors affect oil well productivity?"*
> 
> **COVER THESE CONCEPTS:**
> - **Permeability**: How easily oil flows through rock
> - **Porosity**: How much oil the rock can hold
> - **Pressure**: Driving force for oil production
> - **Well Interference**: How nearby wells affect each other
> - **Drainage Area**: Volume of reservoir each well can access

#### **Physics Behind Well Spacing**
> **DRAW ON BOARD/SLIDE:**
> ```
> Optimal Spacing = f(Reservoir Quality, Economics, Interference)
> 
> Too Close: High interference, reduced recovery per well
> Too Far: Poor drainage, missed opportunities
> Just Right: Maximum economic recovery
> ```

#### **Mathematical Relationships**
> **EXPLAIN:**
> 
> "The productivity of a well follows this relationship:
> 
> **Production Rate = (Permeability × Thickness × Pressure) / (Viscosity × Distance)**
> 
> This is why permeability and pressure are so important in our model!"

---

### **SECTION 3: CODE WALKTHROUGH - DATA GENERATION (30 minutes)**

#### **Section Introduction**
> **SAY TO STUDENTS:**
> 
> "Now let's dive into the code. We'll start by generating synthetic data that mimics real reservoir conditions. This approach is common in petroleum engineering because real data is often proprietary or limited."

#### **Code Walkthrough 1: Class Structure**
```python
class WellSpacingDataGenerator:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
```

> **EXPLAIN:**
> 
> "We use object-oriented programming because it keeps our code organized. This class will generate all the different types of data we need."

#### **Code Walkthrough 2: Reservoir Properties**
```python
def generate_reservoir_properties(self):
    # Permeability (mD) - log-normal distribution
    permeability = np.random.lognormal(mean=np.log(50), sigma=1, size=self.n_samples)
    
    # Porosity (fraction) - normal distribution bounded
    porosity = np.clip(np.random.normal(0.12, 0.03, self.n_samples), 0.05, 0.25)
```

> **PAUSE FOR QUESTIONS**
> 
> **EXPLAIN:**
> - "**Log-normal for permeability**: Permeability varies over orders of magnitude (1 mD to 1000+ mD)"
> - "**Normal for porosity**: Porosity is typically 5-25% with most around 10-15%"
> - "**np.clip()**: Ensures physically realistic values"
> 
> **INTERACTIVE MOMENT:**
> *"Why do you think we use different probability distributions for different rock properties?"*

#### **Code Walkthrough 3: Physics-Based Calculations**
```python
def calculate_production_metrics(self, reservoir_props, well_params, economic_factors):
    # Calculate productivity index
    productivity_index = (reservoir_props['permeability'] * reservoir_props['net_pay'] * 
                        reservoir_props['porosity'] * reservoir_props['oil_saturation']) / 1000
    
    # Calculate interference factor
    interference_factor = 1 - np.exp(-well_params['spacing_to_nearest'] / 800)
```

> **STOP AND EXPLAIN:**
> 
> "This is where petroleum engineering meets data science:
> - **Productivity Index**: Combines all reservoir quality factors
> - **Interference Factor**: Exponential decay - wells interfere less as distance increases
> - **800 ft**: Typical interference distance in shale reservoirs
> 
> **The exponential function models physics**: Close wells (small spacing) have high interference (small factor)"

---

### **SECTION 4: MACHINE LEARNING IMPLEMENTATION (35 minutes)**

#### **Section Introduction**
> **SAY TO STUDENTS:**
> 
> "Now we'll build machine learning models to predict optimal well spacing. We'll compare three different approaches to see which works best."

#### **Code Walkthrough 4: Model Setup**
```python
class WellSpacingOptimizer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'linear_regression': LinearRegression()
        }
```

> **EXPLAIN EACH MODEL:**
> 
> "We're using three different approaches:
> 1. **Random Forest**: Good for non-linear relationships, handles different data types well
> 2. **Gradient Boosting**: Often wins competitions, learns from mistakes iteratively  
> 3. **Linear Regression**: Simple baseline, easy to interpret
> 
> **Why compare multiple models?** Different algorithms work better for different problems!"

#### **Code Walkthrough 5: Feature Preparation**
```python
def prepare_features(self, df):
    feature_columns = [
        'permeability', 'porosity', 'net_pay', 'pressure', 'water_saturation',
        'oil_saturation', 'spacing_to_nearest', 'nearby_wells', 'lateral_length',
        'frac_stages', 'completion_quality', 'drainage_area', 'oil_price',
        'drilling_cost', 'completion_cost', 'operating_cost'
    ]
    
    X = df[feature_columns]  # Input features
    y = df['optimal_spacing']  # Target variable
    
    return X, y, feature_columns
```

> **INTERACTIVE QUESTION:**
> *"Looking at these features, which ones do you think will be most important for predicting optimal well spacing?"*
> 
> **DISCUSS:**
> - Geological factors (permeability, porosity)
> - Engineering factors (nearby wells, completion quality)
> - Economic factors (oil price, costs)

#### **Code Walkthrough 6: Model Training**
```python
def train_models(self, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features for linear regression
    X_train_scaled = self.scaler.fit_transform(X_train)
    X_test_scaled = self.scaler.transform(X_test)
```

> **EXPLAIN IMPORTANT CONCEPTS:**
> 
> "Key machine learning principles here:
> - **Train/Test Split**: Never test on data you trained on!
> - **Feature Scaling**: Linear regression needs features on similar scales
> - **Random State**: Makes results reproducible
> - **80/20 Split**: Common practice - 80% training, 20% testing"

#### **Code Walkthrough 7: Model Evaluation**
```python
for name, model in self.models.items():
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
```

> **EXPLAIN METRICS:**
> 
> "We use multiple metrics to evaluate model performance:
> - **MSE (Mean Squared Error)**: Penalizes large errors heavily
> - **MAE (Mean Absolute Error)**: Average error in feet
> - **R² (R-squared)**: Percentage of variance explained (0-1, higher is better)
> 
> **For well spacing**: MAE tells us average error in feet, R² tells us how much variance we explain"

---

### **SECTION 5: RESULTS ANALYSIS AND VISUALIZATION (25 minutes)**

#### **Code Walkthrough 8: Visualization Function**
```python
def visualize_results(df, model_performance, feature_names, feature_importance):
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Correlation matrix
    plt.subplot(3, 4, 1)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
```

> **EXPLAIN VISUALIZATION STRATEGY:**
> 
> "We create 12 different plots because:
> 1. **Correlation Matrix**: Shows relationships between variables
> 2. **Distribution Plots**: Understanding data patterns
> 3. **Scatter Plots**: Revealing trends and outliers
> 4. **Performance Comparison**: Model selection
> 5. **Feature Importance**: Understanding what drives predictions
> 
> **Professional Tip**: Always create multiple views of your data!"

#### **Interactive Analysis Session**
> **SHOW STUDENTS THE PLOTS AND ASK:**
> 
> 1. *"Looking at the correlation matrix, what relationships do you notice?"*
> 2. *"Which model performed best and why?"*
> 3. *"What are the most important features for predicting optimal spacing?"*
> 4. *"Do the results make sense from a petroleum engineering perspective?"*

---

### **SECTION 6: BUSINESS APPLICATIONS (20 minutes)**

#### **Real-World Implementation**
> **DISCUSS WITH STUDENTS:**
> 
> "How would you implement this in a real company?
> 
> **Phase 1: Data Collection**
> - Gather historical well data
> - Collect reservoir characterization
> - Compile economic parameters
> 
> **Phase 2: Model Validation**
> - Test predictions against actual results
> - Adjust for local conditions
> - Incorporate regulatory constraints
> 
> **Phase 3: Decision Support**
> - Create user-friendly interface
> - Integrate with existing workflows
> - Provide uncertainty estimates"

#### **Economic Impact Analysis**
> **CALCULATE WITH STUDENTS:**
> 
> "Let's estimate the value of this optimization:
> - **Scenario**: 100-well drilling program
> - **Current Practice**: Fixed 800-ft spacing
> - **Optimized Practice**: ML-recommended spacing
> - **Improvement**: 5% better NPV per well
> - **Value**: $2M × 100 wells × 5% = $10M improvement
> 
> **ROI on ML Project**: Development cost ~$100K, return $10M = 100:1 ROI!"

---

### **SECTION 7: HANDS-ON CODING SESSION (30 minutes)**

#### **Guided Practice**
> **TELL STUDENTS:**
> 
> "Now you'll run the code yourself. Follow along as we execute each section."

#### **Step-by-Step Execution**
1. **Run Data Generation**
   ```python
   data_generator = WellSpacingDataGenerator(n_samples=1500)
   df = data_generator.generate_complete_dataset()
   print(df.head())
   ```

2. **Examine the Data**
   ```python
   print(df.describe())
   print(df.info())
   ```

3. **Train Models**
   ```python
   optimizer = WellSpacingOptimizer()
   X, y, feature_names = optimizer.prepare_features(df)
   model_performance, X_test, y_test = optimizer.train_models(X, y)
   ```

4. **Analyze Results**
   ```python
   # Students examine model performance
   for model, metrics in model_performance.items():
       print(f"{model}: R² = {metrics['r2']:.3f}")
   ```

#### **Individual Exercise**
> **ASSIGN STUDENTS:**
> 
> "Modify the code to:
> 1. Change the number of samples to 500
> 2. Add a new feature (e.g., 'rock_type')
> 3. Try different model parameters
> 4. Create one additional visualization
> 
> **Time**: 15 minutes
> **Goal**: Understand how changes affect results"

---

### **SECTION 8: ADVANCED TOPICS AND EXTENSIONS (15 minutes)**

#### **Limitations Discussion**
> **FACILITATE DISCUSSION:**
> 
> "What are the limitations of our approach?
> - **Synthetic Data**: Real reservoirs are more complex
> - **Static Model**: Reservoir properties change over time
> - **Economic Assumptions**: Oil prices fluctuate
> - **Regulatory Constraints**: Not included in model
> - **Geological Uncertainty**: Perfect knowledge assumed"

#### **Advanced Extensions**
> **PRESENT ADVANCED CONCEPTS:**
> 
> "How could we improve this project?
> 
> **Technical Improvements:**
> - **Deep Learning**: Neural networks for complex patterns
> - **Ensemble Methods**: Combine multiple models
> - **Time Series**: Include production decline curves
> - **Uncertainty Quantification**: Probabilistic predictions
> 
> **Business Improvements:**
> - **Real-Time Updates**: Incorporate new data
> - **Sensitivity Analysis**: What-if scenarios
> - **Optimization Algorithms**: Find global optimum
> - **Risk Assessment**: Quantify decision uncertainty"

---

### **SECTION 9: WRAP-UP AND ASSESSMENT (10 minutes)**

#### **Key Takeaways Review**
> **SUMMARIZE WITH STUDENTS:**
> 
> "What did we learn today?
> 1. **Problem Definition**: Well spacing is a complex optimization problem
> 2. **Data Generation**: Synthetic data can model real physics
> 3. **Model Comparison**: Different algorithms have different strengths
> 4. **Feature Importance**: Understanding what drives predictions
> 5. **Business Value**: ML can create significant economic value
> 6. **Visualization**: Multiple views reveal different insights"

#### **Quick Assessment Questions**
> **ASK STUDENTS:**
> 
> 1. "Why do we use log-normal distribution for permeability?"
> 2. "What does R² = 0.85 mean in practical terms?"
> 3. "How would you explain the business value to a non-technical manager?"
> 4. "What additional data would improve our model?"

#### **Next Steps**
> **ASSIGN FOR NEXT CLASS:**
> 
> "**Homework Assignment:**
> 1. Research one real-world well spacing optimization case study
> 2. Identify three additional features we could include
> 3. Propose one improvement to our visualization approach
> 4. Write a one-page executive summary of our results
> 
> **Due**: Next class session"

---

## **Teaching Tips and Best Practices**

### **Engagement Strategies**
- **Start with questions**: Get students thinking before presenting concepts
- **Use analogies**: Compare well interference to traffic congestion
- **Real examples**: Show actual well spacing maps from published studies
- **Interactive coding**: Students type along during demonstrations
- **Peer discussion**: Pair students for analysis discussions

### **Common Student Questions and Answers**

**Q: "Why not just use rules of thumb for well spacing?"**
A: "Rules of thumb work for average conditions, but every reservoir is different. ML helps us customize decisions for specific conditions, potentially worth millions in improved recovery."

**Q: "How do we know if our synthetic data is realistic?"**
A: "We base distributions on published literature and real field data. The key is that relationships between variables follow known physics, even if individual values are synthetic."

**Q: "Which model should we use in practice?"**
A: "Depends on your priorities. Random Forest is often best for accuracy and interpretability. Linear regression is good when you need to explain every coefficient to regulators."

**Q: "What if the model is wrong?"**
A: "All models are wrong, but some are useful. We use cross-validation, multiple metrics, and business sense to minimize risk. The goal is better decisions, not perfect predictions."

### **Troubleshooting Common Issues**

1. **Code doesn't run**: Check Python environment, install missing packages
2. **Plots look different**: Matplotlib/seaborn version differences
3. **Students confused by statistics**: Provide quick refresher on correlation, R²
4. **Petroleum engineering concepts unclear**: Use more analogies and visual aids

### **Assessment Rubric**

| Criteria | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|----------|---------------|----------|------------------|----------------------|
| **Technical Understanding** | Explains all concepts clearly | Understands most concepts | Basic understanding | Significant gaps |
| **Code Execution** | Runs code independently | Runs with minimal help | Needs some guidance | Requires significant help |
| **Data Interpretation** | Insightful analysis | Good interpretation | Basic analysis | Minimal interpretation |
| **Business Application** | Clear value proposition | Understands business impact | Basic business connection | Limited business understanding |

---

## **Additional Resources**

### **Recommended Reading**
- "Petroleum Engineering Handbook" - SPE
- "Hands-On Machine Learning" - Aurélien Géron
- "Data Science for Business" - Provost & Fawcett

### **Online Resources**
- SPE (Society of Petroleum Engineers) papers on well spacing
- Kaggle petroleum engineering datasets
- scikit-learn documentation and tutorials

### **Software Requirements**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### **Dataset Sources for Future Projects**
- Texas Railroad Commission well data
- North Dakota Industrial Commission data
- Colorado Oil and Gas Conservation Commission data

---

*This teaching plan provides a comprehensive framework for delivering an engaging and educational session on well spacing optimization using machine learning. Adjust timing and depth based on your specific audience and available time.*