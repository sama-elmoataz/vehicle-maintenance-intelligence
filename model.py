from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# LOAD DATA
# =========================================================

BASE_DIR=Path(__file__).resolve().parent

DATA_CANDIDATES=[
    BASE_DIR/"logistics_sample_15k (1).csv",
    BASE_DIR/"logistics_sample_15k.csv"
]

DATA_PATH=None

for candidate in DATA_CANDIDATES:
    if candidate.exists():
        DATA_PATH=candidate
        break

if DATA_PATH is None:
    raise FileNotFoundError(
        "Dataset not found. Put either "
        "'logistics_sample_15k (1).csv' or "
        "'logistics_sample_15k.csv' in the same folder as model.py."
    )

df=pd.read_csv(DATA_PATH)


# =========================================================
# TARGET AND FEATURES
# =========================================================

y=df['Maintenance_Level_Code']

# Final model decision:
# Severity_Score and Predictive_Score are intentionally excluded
# from training and prediction.
x=df.drop(columns=[
    'Maintenance_Required',
    'Maintenance_Level',
    'Maintenance_Level_Code',
    'Severity_Score',
    'Predictive_Score'
]).copy()


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

x_train=x_train.copy()
x_test=x_test.copy()


# =========================================================
# PREPROCESSING
# =========================================================

# Vibration Levels
x_train['Vibration_Levels']=x_train['Vibration_Levels'].clip(lower=0)
x_test['Vibration_Levels']=x_test['Vibration_Levels'].clip(lower=0)

x_train['Vibration_Levels']=np.log1p(x_train['Vibration_Levels'])
x_test['Vibration_Levels']=np.log1p(x_test['Vibration_Levels'])


# Oil Quality - IQR capping based on training data
q25=x_train['Oil_Quality'].quantile(0.25)
q75=x_train['Oil_Quality'].quantile(0.75)
iqr=q75-q25

lower_bound=q25-(1.5*iqr)
upper_bound=q75+(1.5*iqr)

x_train['Oil_Quality']=np.where(
    x_train['Oil_Quality']>upper_bound,
    upper_bound,
    x_train['Oil_Quality']
)

x_train['Oil_Quality']=np.where(
    x_train['Oil_Quality']<lower_bound,
    lower_bound,
    x_train['Oil_Quality']
)

x_test['Oil_Quality']=np.where(
    x_test['Oil_Quality']>upper_bound,
    upper_bound,
    x_test['Oil_Quality']
)

x_test['Oil_Quality']=np.where(
    x_test['Oil_Quality']<lower_bound,
    lower_bound,
    x_test['Oil_Quality']
)

x_train['Oil_Quality']=x_train['Oil_Quality'].clip(upper=100.0)
x_test['Oil_Quality']=x_test['Oil_Quality'].clip(upper=100.0)


# Brake Condition
brake_mapping={
    'Poor':0,
    'Fair':1,
    'Good':2
}

x_train['Brake_Condition']=x_train['Brake_Condition'].map(brake_mapping)
x_test['Brake_Condition']=x_test['Brake_Condition'].map(brake_mapping)


# Failure History
x_train['Failure_History']=(
    x_train['Failure_History']>=0.5
).astype(int)

x_test['Failure_History']=(
    x_test['Failure_History']>=0.5
).astype(int)


# Anomalies Detected
x_train['Anomalies_Detected']=(
    x_train['Anomalies_Detected']>=0.5
).astype(int)

x_test['Anomalies_Detected']=(
    x_test['Anomalies_Detected']>=0.5
).astype(int)


# Weather Conditions
weather_mapping={
    'Clear':0,
    'Rainy':1,
    'Snowy':2,
    'Windy':3
}

x_train['Weather_Conditions']=x_train['Weather_Conditions'].map(
    weather_mapping
)

x_test['Weather_Conditions']=x_test['Weather_Conditions'].map(
    weather_mapping
)


# Road Conditions
road_mapping={
    'Urban':0,
    'Rural':1,
    'Highway':2
}

x_train['Road_Conditions']=x_train['Road_Conditions'].map(
    road_mapping
)

x_test['Road_Conditions']=x_test['Road_Conditions'].map(
    road_mapping
)


# Delivery Times
x_train['Delivery_Times']=np.log1p(
    x_train['Delivery_Times']
)

x_test['Delivery_Times']=np.log1p(
    x_test['Delivery_Times']
)


# Downtime Maintenance
x_train['Downtime_Maintenance']=(
    x_train['Downtime_Maintenance']
    .clip(lower=0.0)
)

x_test['Downtime_Maintenance']=(
    x_test['Downtime_Maintenance']
    .clip(lower=0.0)
)

x_train['Downtime_Maintenance']=np.log1p(
    x_train['Downtime_Maintenance']
)

x_test['Downtime_Maintenance']=np.log1p(
    x_test['Downtime_Maintenance']
)


# Impact on Efficiency
x_train['Impact_on_Efficiency']=np.log1p(
    x_train['Impact_on_Efficiency']
)

x_test['Impact_on_Efficiency']=np.log1p(
    x_test['Impact_on_Efficiency']
)


# Vehicle ID
x_train=x_train.drop(columns=['Vehicle_ID'])
x_test=x_test.drop(columns=['Vehicle_ID'])


# Make and Model - Target Encoding
# Fit the encoder only on the Make_and_Model column.
# This keeps the encoder independent from the number of other model features.
target_encoder=TargetEncoder(
    cols=['Make_and_Model'],
    handle_missing='value',
    handle_unknown='value'
)

make_model_train=target_encoder.fit_transform(
    x_train[['Make_and_Model']],
    y_train
)

make_model_test=target_encoder.transform(
    x_test[['Make_and_Model']]
)

x_train['Make_and_Model_Encoded']=make_model_train['Make_and_Model']
x_test['Make_and_Model_Encoded']=make_model_test['Make_and_Model']

x_train=x_train.drop(
    columns=['Make_and_Model']
)

x_test=x_test.drop(
    columns=['Make_and_Model']
)


# Vehicle Age
current_year=2026

x_train['Vehicle_Age']=(
    current_year-x_train['Year_of_Manufacture']
)

x_test['Vehicle_Age']=(
    current_year-x_test['Year_of_Manufacture']
)

x_train=x_train.drop(
    columns=['Year_of_Manufacture']
)

x_test=x_test.drop(
    columns=['Year_of_Manufacture']
)


# Vehicle Type
vehicle_type_mapping={
    'Van':0,
    'Truck':1
}

x_train['Vehicle_Type']=x_train['Vehicle_Type'].map(
    vehicle_type_mapping
)

x_test['Vehicle_Type']=x_test['Vehicle_Type'].map(
    vehicle_type_mapping
)


# Usage Hours
x_train['Usage_Hours']=(
    x_train['Usage_Hours']
    .clip(lower=0)
)

x_test['Usage_Hours']=(
    x_test['Usage_Hours']
    .clip(lower=0)
)

x_train['Usage_Hours']=np.log1p(
    x_train['Usage_Hours']
)

x_test['Usage_Hours']=np.log1p(
    x_test['Usage_Hours']
)


# Route Info
x_train=x_train.drop(columns=['Route_Info'])
x_test=x_test.drop(columns=['Route_Info'])


# Load Ratio
x_train['capacity_ratio']=(
    x_train['Actual_Load']/
    x_train['Load_Capacity']
)

x_test['capacity_ratio']=(
    x_test['Actual_Load']/
    x_test['Load_Capacity']
)

x_train=x_train.drop(
    columns=[
        'Load_Capacity',
        'Actual_Load'
    ]
)

x_test=x_test.drop(
    columns=[
        'Load_Capacity',
        'Actual_Load'
    ]
)


# Last Maintenance Date
x_train['Last_Maintenance_Date']=pd.to_datetime(
    x_train['Last_Maintenance_Date']
)

x_test['Last_Maintenance_Date']=pd.to_datetime(
    x_test['Last_Maintenance_Date']
)

reference_date=pd.Timestamp('2026-01-01')

x_train['Days_Since_Maintenance']=(
    reference_date-
    x_train['Last_Maintenance_Date']
).dt.days

x_test['Days_Since_Maintenance']=(
    reference_date-
    x_test['Last_Maintenance_Date']
).dt.days

x_train=x_train.drop(
    columns=['Last_Maintenance_Date']
)

x_test=x_test.drop(
    columns=['Last_Maintenance_Date']
)


# Maintenance Type - One Hot Encoding
one_hot_encoder=OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

maintenance_train=one_hot_encoder.fit_transform(
    x_train[['Maintenance_Type']]
)

maintenance_test=one_hot_encoder.transform(
    x_test[['Maintenance_Type']]
)

encoded_cols=one_hot_encoder.get_feature_names_out(
    ['Maintenance_Type']
)

maintenance_train_df=pd.DataFrame(
    maintenance_train,
    columns=encoded_cols,
    index=x_train.index
)

maintenance_test_df=pd.DataFrame(
    maintenance_test,
    columns=encoded_cols,
    index=x_test.index
)

x_train=pd.concat(
    [
        x_train.drop(
            columns=['Maintenance_Type']
        ),
        maintenance_train_df
    ],
    axis=1
)

x_test=pd.concat(
    [
        x_test.drop(
            columns=['Maintenance_Type']
        ),
        maintenance_test_df
    ],
    axis=1
)


# Maintenance Cost
x_train['Maintenance_Cost']=np.log1p(
    x_train['Maintenance_Cost']
)

x_test['Maintenance_Cost']=np.log1p(
    x_test['Maintenance_Cost']
)


# Engine Temperature
x_train=x_train.drop(
    columns=['Engine_Temperature']
)

x_test=x_test.drop(
    columns=['Engine_Temperature']
)


# Tire Pressure
x_train['Tire_Pressure']=np.log1p(
    x_train['Tire_Pressure']
)

x_test['Tire_Pressure']=np.log1p(
    x_test['Tire_Pressure']
)


# Fuel Consumption
x_train['Fuel_Consumption']=np.log1p(
    x_train['Fuel_Consumption']
)

x_test['Fuel_Consumption']=np.log1p(
    x_test['Fuel_Consumption']
)


# Battery Status
x_train['Is_Battery_Elevated']=(
    x_train['Battery_Status']>47.5
).astype(int)

x_test['Is_Battery_Elevated']=(
    x_test['Battery_Status']>47.5
).astype(int)

x_train=x_train.drop(
    columns=['Battery_Status']
)

x_test=x_test.drop(
    columns=['Battery_Status']
)


# Overload Indicator
x_train['Is_Overloaded']=(
    x_train['capacity_ratio']>1.0
).astype(int)

x_test['Is_Overloaded']=(
    x_test['capacity_ratio']>1.0
).astype(int)


# =========================================================
# FEATURE SCALING
# =========================================================

scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(
    x_train
)

x_test_scaled=scaler.transform(
    x_test
)


# =========================================================
# MODELS
# =========================================================

svm=Pipeline([
    (
        'scaler',
        StandardScaler()
    ),
    (
        'svm',
        SVC(
            kernel='rbf',
            C=1,
            probability=True,
            random_state=42
        )
    )
])

rf=RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

et=ExtraTreesClassifier(
    n_estimators=200,
    random_state=42
)

voting=VotingClassifier(
    estimators=[
        ('svm',svm),
        ('rf',rf),
        ('et',et)
    ],
    voting='soft'
)

knn=KNeighborsClassifier(
    n_neighbors=5
)

logistic=LogisticRegression(
    max_iter=1000,
    random_state=42
)


# =========================================================
# TRAIN MODELS
# =========================================================

svm.fit(
    x_train,
    y_train
)

rf.fit(
    x_train,
    y_train
)

et.fit(
    x_train,
    y_train
)

voting.fit(
    x_train,
    y_train
)

knn.fit(
    x_train_scaled,
    y_train
)

logistic.fit(
    x_train_scaled,
    y_train
)


# =========================================================
# PREDICTIONS
# =========================================================

y_pred_svm=svm.predict(
    x_test
)

y_pred_rf=rf.predict(
    x_test
)

y_pred_et=et.predict(
    x_test
)

y_pred_voting=voting.predict(
    x_test
)

y_pred_knn=knn.predict(
    x_test_scaled
)

y_pred_logistic=logistic.predict(
    x_test_scaled
)


# =========================================================
# MODEL COMPARISON
# =========================================================

models={
    'SVM':y_pred_svm,
    'Random Forest':y_pred_rf,
    'Extra Trees':y_pred_et,
    'Voting Classifier':y_pred_voting,
    'KNN':y_pred_knn,
    'Logistic Regression':y_pred_logistic
}

results=[]

for model_name,predictions in models.items():

    accuracy=accuracy_score(
        y_test,
        predictions
    )

    precision=precision_score(
        y_test,
        predictions,
        average='weighted',
        zero_division=0
    )

    recall=recall_score(
        y_test,
        predictions,
        average='weighted',
        zero_division=0
    )

    f1=f1_score(
        y_test,
        predictions,
        average='weighted',
        zero_division=0
    )

    results.append([
        model_name,
        accuracy,
        precision,
        recall,
        f1
    ])

results_df=pd.DataFrame(
    results,
    columns=[
        'Model',
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score'
    ]
)

results_df=results_df.sort_values(
    by='Accuracy',
    ascending=False
).reset_index(drop=True)
