# Vehicle Maintenance Prediction

A machine learning project for predicting vehicle maintenance requirements using vehicle condition, maintenance history, usage, and operational data.

The model classifies each vehicle into three maintenance levels:

- **Normal**
- **Minor**
- **Major**

The machine learning workflow is implemented in `model.py`, while `app.py` provides an interactive Streamlit interface for testing predictions and maintenance planning.

---

## Project Objective

The target variable is:

```text
Maintenance_Level_Code
```

Target-related columns are excluded from the model input:

```text
Maintenance_Required
Maintenance_Level
Maintenance_Level_Code
```

The project is therefore a **multiclass classification problem**.

---

## Train/Test Split

The dataset is divided into:

- **80% Training**
- **20% Testing**

using:

```python
train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y` helps preserve the maintenance-class distribution in both datasets.

---

# Data Preprocessing

The preprocessing workflow handles each feature based on its data type and distribution.

Main steps include:

- Negative-value clipping
- Log transformations
- IQR-based outlier handling
- Binary encoding
- Ordinal mapping
- Target Encoding
- One-Hot Encoding
- Feature scaling
- Feature engineering

### Log-transformed features

`np.log1p()` is applied to several skewed variables, including:

```text
Vibration_Levels
Predictive_Score
Usage_Hours
Maintenance_Cost
Tire_Pressure
Fuel_Consumption
Delivery_Times
Downtime_Maintenance
Impact_on_Efficiency
Severity_Score
```

### Oil Quality

Oil Quality outliers are capped using the **IQR method** calculated from the training data.

The feature is also capped at a maximum value of `100`.

---

## Feature Encoding

### Brake Condition

```text
Poor = 0
Fair = 1
Good = 2
```

### Vehicle Type

```text
Van = 0
Truck = 1
```

### Weather Conditions

```text
Clear = 0
Rainy = 1
Snowy = 2
Windy = 3
```

### Road Conditions

```text
Urban = 0
Rural = 1
Highway = 2
```

### Binary Features

`Failure_History` and `Anomalies_Detected` are converted into binary variables using a threshold of `0.5`.

---

# Feature Engineering

Several additional variables are created before model training.

### Vehicle Age

```python
Vehicle_Age = 2026 - Year_of_Manufacture
```

### Load Ratio

```python
capacity_ratio = Actual_Load / Load_Capacity
```

A ratio above `1.0` indicates an overloaded vehicle.

```python
Is_Overloaded = capacity_ratio > 1.0
```

### Days Since Maintenance

```python
reference_date = pd.Timestamp('2026-01-01')

Days_Since_Maintenance = (
    reference_date - Last_Maintenance_Date
).dt.days
```

### Vehicle Make and Model

`Make_and_Model` is transformed using **Target Encoding**.

The resulting feature is:

```text
Make_and_Model_Encoded
```

### Maintenance Type

Maintenance Type is transformed using One-Hot Encoding:

```python
OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)
```

### Battery Status

A binary battery feature is created:

```python
Is_Battery_Elevated = Battery_Status > 47.5
```

---

## Removed Features

Some original variables are removed after preprocessing or replaced by engineered features:

```text
Vehicle_ID
Route_Info
Year_of_Manufacture
Load_Capacity
Actual_Load
Last_Maintenance_Date
Engine_Temperature
Battery_Status
```

---

# Feature Scaling

`StandardScaler` is used for models that are sensitive to feature magnitude, including:

- Support Vector Machine
- K-Nearest Neighbors
- Logistic Regression

The SVM model includes scaling inside a pipeline:

```python
svm=Pipeline([
    ('scaler',StandardScaler()),
    ('svm',SVC(
        kernel='rbf',
        C=1,
        probability=True,
        random_state=42
    ))
])
```

---

# Models Compared

Six classification approaches are evaluated:

1. Support Vector Machine
2. Random Forest
3. Extra Trees
4. Soft Voting Classifier
5. K-Nearest Neighbors
6. Logistic Regression

### Random Forest

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

### Extra Trees

```python
ExtraTreesClassifier(
    n_estimators=200,
    random_state=42
)
```

### KNN

```python
KNeighborsClassifier(
    n_neighbors=5
)
```

### Logistic Regression

```python
LogisticRegression(
    max_iter=1000
)
```

The Soft Voting Classifier combines:

```text
SVM + Random Forest + Extra Trees
```

using predicted probabilities.

---

# Model Evaluation

Models are compared using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

Additional evaluation includes:

- Classification Report
- Confusion Matrix
- Feature Importance
- Model Comparison

The **Random Forest Classifier** is used as the primary prediction engine in the final application.

---

## Random Forest Feature Importance

Feature importance is used to understand the model's overall behavior.

Some of the strongest features identified include:

```text
Severity Score
Anomalies Detected
Brake Condition
Failure History
Fuel Consumption
Downtime Maintenance
Maintenance Cost
Predictive Score
```

Feature importance represents global model behavior and does not imply causation.

---

# Machine Learning Workflow

```text
Raw Dataset
     ↓
Train / Test Split
     ↓
Data Cleaning
     ↓
Outlier Handling
     ↓
Encoding
     ↓
Feature Engineering
     ↓
Feature Scaling
     ↓
Train Multiple Models
     ↓
Model Evaluation
     ↓
Random Forest
     ↓
Normal / Minor / Major
```

The same preprocessing logic is applied to new vehicle inputs before prediction.

---

# Streamlit Application

The Streamlit application provides an interactive layer on top of the machine learning model.

## Dashboard

Provides a fleet-level overview including:

- Maintenance-level distribution
- Anomalies detected
- Poor brake condition rate
- Overloaded vehicle rate
- Days since maintenance
- Vehicle-type analysis
- Operational insights

## Predict Maintenance

Users can enter vehicle information and receive:

- Predicted Maintenance Level
- Prediction Confidence
- Probability Breakdown
- Risk Score
- Risk Gauge
- Vehicle Condition Snapshot
- Maintenance Priority
- Downloadable PDF Report

## Maintenance Planning

Users can:

- Schedule maintenance
- Select maintenance type
- Set priority
- Add technician or workshop details
- Estimate maintenance cost
- Save maintenance plans during the current session

## Model Performance & Data Insights

Includes:

- Model comparison
- Accuracy, Precision, Recall and F1
- Random Forest confusion matrix
- Feature importance
- Vehicle-type analysis
- Usage analysis

---

# Maintenance Risk Score

The application converts model probabilities into a simplified score:

```text
Normal = 0
Minor = 50
Major = 100
```

Conceptually:

```python
Risk Score =
P(Normal) * 0 +
P(Minor) * 50 +
P(Major) * 100
```

This is an interface-level interpretation of the model probabilities and is **not an engineering risk measurement**.

---

# Maintenance Cost Planning

Maintenance cost estimates are based on historical records.

Priority applies a planning multiplier:

| Priority | Multiplier |
|---|---:|
| Routine | 1.00x |
| Medium | 1.10x |
| High | 1.25x |
| Critical | 1.45x |

These multipliers are planning assumptions and are not learned by the machine learning model.

---

# Project Structure

GitHub repository:

```text
vehicle-maintenance-intelligence/
├── app.py
├── model.py
├── requirements.txt
├── README.md
└── .gitignore
```

Local project:

```text
vehicle-maintenance-intelligence/
├── app.py
├── model.py
├── logistics_sample_15k (1).csv
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset is intentionally excluded from GitHub.

---

# Dataset

The project expects:

```text
logistics_sample_15k (1).csv
```

The CSV is excluded through `.gitignore` because its redistribution status has not been confirmed.

Place the dataset locally in the project root before running the project.

---

# Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# Technologies

- Python
- pandas
- NumPy
- Scikit-Learn
- Category Encoders
- Matplotlib
- Seaborn
- Streamlit
- Git
- GitHub

---

# Limitations

- Model results depend on patterns in the current dataset.
- Very high model performance should be validated using independent unseen data.
- Feature importance does not imply causation.
- The UI risk score is not an engineering risk measurement.
- Vehicle Condition Snapshot indicators are descriptive rather than diagnostic.
- Maintenance cost values are planning estimates.
- Priority multipliers are manually defined.
- Saved maintenance plans last only during the active Streamlit session.

---

# Future Improvements

- External model validation
- Model serialization
- Persistent database storage
- Automated preprocessing pipelines
- Work-order management
- User authentication
- Real-time telematics integration
- Model monitoring and retraining
- API deployment
- Cloud deployment

---

# Disclaimer

This project was developed for machine learning and predictive-maintenance analysis.

Its predictions are intended to support maintenance planning and should not replace professional vehicle inspection or safety-critical engineering decisions.