import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from IPython.display import display


df=pd.read_csv("logistics_sample_15k (1).csv")


df.isnull().sum()

print(df.head())

df.duplicated().sum()


y = df['Maintenance_Level_Code']
x = df.drop(columns=[
        'Maintenance_Required',
        'Maintenance_Level',
        'Maintenance_Level_Code',
    ]
)


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=42,stratify=y)


print(x_train['Vibration_Levels'].head())

print(x_train['Vibration_Levels'].describe())

x_train['Vibration_Levels'] = x_train['Vibration_Levels'].clip(lower=0)
x_test['Vibration_Levels'] = x_test['Vibration_Levels'].clip(lower=0)

print(x_train['Vibration_Levels'].describe())

sns.boxplot(x='Maintenance_Required', y='Vibration_Levels', data=df)
plt.xlabel('Maintenance Required')
plt.ylabel('Vibration Levels')
plt.title('Vibration Levels vs Maintenance Required')
plt.show()

sns.histplot(x_train['Vibration_Levels'], kde=True)
plt.show()

x_train['Vibration_Levels'] = np.log1p(x_train['Vibration_Levels'])
x_test['Vibration_Levels'] = np.log1p(x_test['Vibration_Levels'])

sns.histplot(x_train['Vibration_Levels'], kde=True)
plt.show()

print(x_train['Vibration_Levels'].describe())


print(x_train['Oil_Quality'].describe())

q25 = x_train['Oil_Quality'].quantile(0.25)
q75 = x_train['Oil_Quality'].quantile(0.75)
iqr = q75 - q25
lower_bound = q25 - (1.5 * iqr)
upper_bound = q75 + (1.5 * iqr)
x_train['Oil_Quality'] = np.where(x_train['Oil_Quality'] > upper_bound, upper_bound, x_train['Oil_Quality'])
x_train['Oil_Quality'] = np.where(x_train['Oil_Quality'] < lower_bound, lower_bound, x_train['Oil_Quality'])
x_test['Oil_Quality']= np.where(x_test['Oil_Quality'] > upper_bound, upper_bound, x_test['Oil_Quality'])
x_test['Oil_Quality']= np.where(x_test['Oil_Quality'] < lower_bound, lower_bound, x_test['Oil_Quality'])

x_train['Oil_Quality'] = x_train['Oil_Quality'].clip(upper=100.0)
x_test['Oil_Quality'] = x_test['Oil_Quality'].clip(upper=100.0)

print(x_train['Oil_Quality'].describe())

sns.histplot(x_train['Oil_Quality'], kde=True)
plt.show()


brake_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2}
x_train['Brake_Condition'] = x_train['Brake_Condition'].map(brake_mapping)
x_test['Brake_Condition'] = x_test['Brake_Condition'].map(brake_mapping)


print(x_train['Failure_History'].describe())

sns.histplot(x_train['Failure_History'], kde=True)
plt.show()

x_train['Failure_History'] = (x_train['Failure_History'] >= 0.5).astype(int)
x_test['Failure_History'] = (x_test['Failure_History'] >= 0.5).astype(int)

print(x_train['Failure_History'].describe())


print(x_train['Anomalies_Detected'].describe())

sns.histplot(x_train['Anomalies_Detected'], kde=True)
plt.show()

x_train['Anomalies_Detected'] = (x_train['Anomalies_Detected'] >= 0.5).astype(int)
x_test['Anomalies_Detected'] = (x_test['Anomalies_Detected'] >= 0.5).astype(int)


print(x_train['Predictive_Score'].describe())

sns.histplot(x_train['Predictive_Score'], kde=True)
plt.show()

x_train['Predictive_Score'] = x_train['Predictive_Score'].clip(lower=0.0, upper=1.0)
x_test['Predictive_Score'] = x_test['Predictive_Score'].clip(lower=0.0, upper=1.0)

x_train['Predictive_Score'] = np.log1p(x_train['Predictive_Score'])
x_test['Predictive_Score'] = np.log1p(x_test['Predictive_Score'])

sns.histplot(x_train['Predictive_Score'], kde=True)
plt.show()


print(df['Weather_Conditions'].unique())

Weather_Conditions = {
    'Clear': 0,
    'Rainy': 1,
    'Snowy': 2,
    'Windy': 3
}
x_train['Weather_Conditions'] = x_train['Weather_Conditions'].map(Weather_Conditions)
x_test['Weather_Conditions'] = x_test['Weather_Conditions'].map(Weather_Conditions)


Road_Conditions = {'Urban': 0, 'Rural': 1, 'Highway': 2}
x_train['Road_Conditions'] = x_train['Road_Conditions'].map(Road_Conditions)
x_test['Road_Conditions'] = x_test['Road_Conditions'].map(Road_Conditions)


print(x_train['Delivery_Times'].describe())

sns.histplot(x_train['Delivery_Times'], kde=True)
plt.show()

x_train['Delivery_Times'] = np.log1p(x_train['Delivery_Times'])
x_test['Delivery_Times'] = np.log1p(x_test['Delivery_Times'])

sns.histplot(x_train['Delivery_Times'], kde=True)
plt.show()


print(x_train['Downtime_Maintenance'].describe())

sns.histplot(x_train['Downtime_Maintenance'], kde=True)
plt.show()

x_train['Downtime_Maintenance'] = x_train['Downtime_Maintenance'].clip(lower=0.0)
x_test['Downtime_Maintenance'] = x_test['Downtime_Maintenance'].clip(lower=0.0)

x_train['Downtime_Maintenance'] = np.log1p(x_train['Downtime_Maintenance'])
x_test['Downtime_Maintenance'] = np.log1p(x_test['Downtime_Maintenance'])

sns.histplot(x_train['Downtime_Maintenance'], kde=True)
plt.show()


print(x_train['Impact_on_Efficiency'].describe())

sns.histplot(x_train['Impact_on_Efficiency'], kde=True)
plt.show()

x_train['Impact_on_Efficiency'] = np.log1p(x_train['Impact_on_Efficiency'])
x_test['Impact_on_Efficiency'] = np.log1p(x_test['Impact_on_Efficiency'])

sns.histplot(x_train['Impact_on_Efficiency'], kde=True)
plt.show()


print(x_train['Severity_Score'].describe())

sns.histplot(x_train['Severity_Score'], kde=True)
plt.show()

print(x_test['Severity_Score'].describe())

x_train['Severity_Score'] = np.log1p(x_train['Severity_Score'])
x_test['Severity_Score'] = np.log1p(x_test['Severity_Score'])

sns.histplot(x_train['Severity_Score'], kde=True)
plt.show()


x_train = x_train.drop(columns=['Vehicle_ID'])
x_test = x_test.drop(columns=['Vehicle_ID'])


target_encoder = TargetEncoder(
    cols=['Make_and_Model'],
    handle_missing='value',
    handle_unknown='value'
)
x_train = target_encoder.fit_transform(
    x_train,
    y_train
)
x_test = target_encoder.transform(
    x_test
)

x_train.head()

x_train = x_train.rename(columns={'Make_and_Model': 'Make_and_Model_Encoded'})
x_test = x_test.rename(columns={'Make_and_Model': 'Make_and_Model_Encoded'})


current_year = 2026

x_train['Vehicle_Age'] = (
    current_year - x_train['Year_of_Manufacture']
)

x_train = x_train.drop(columns=['Year_of_Manufacture'])

x_test['Vehicle_Age'] = (
    current_year - x_test['Year_of_Manufacture']
)

x_test = x_test.drop(columns=['Year_of_Manufacture'])


vehicle_type = {'Van': 0, 'Truck': 1}
x_train['Vehicle_Type'] = x_train['Vehicle_Type'].map(vehicle_type)
x_test['Vehicle_Type'] = x_test['Vehicle_Type'].map(vehicle_type)


sns.histplot(x_train['Usage_Hours'], kde=True)
plt.show()

print(df['Usage_Hours'].describe())

x_train['Usage_Hours'] = x_train['Usage_Hours'].clip(lower=0)
x_test['Usage_Hours'] = x_test['Usage_Hours'].clip(lower=0)

x_train['Usage_Hours'] = np.log1p(x_train['Usage_Hours'])
x_test['Usage_Hours'] = np.log1p(x_test['Usage_Hours'])

sns.histplot(x_train['Usage_Hours'], kde=True)
plt.show()


x_train=x_train.drop(columns=['Route_Info'])
x_test=x_test.drop(columns=['Route_Info'])


x_train['capacity_ratio']=x_train['Actual_Load']/x_train['Load_Capacity']
x_test['capacity_ratio']=x_test['Actual_Load']/x_test['Load_Capacity']
x_train=x_train.drop(columns=['Load_Capacity','Actual_Load'])
x_test=x_test.drop(columns=['Load_Capacity','Actual_Load'])


x_train['Last_Maintenance_Date'] = pd.to_datetime(x_train['Last_Maintenance_Date'])
x_test['Last_Maintenance_Date'] = pd.to_datetime(x_test['Last_Maintenance_Date'])

reference_date = pd.Timestamp('2026-01-01')
x_train['Days_Since_Maintenance'] = (reference_date - x_train['Last_Maintenance_Date']).dt.days
x_test['Days_Since_Maintenance'] = (reference_date - x_test['Last_Maintenance_Date']).dt.days

x_train = x_train.drop(columns=['Last_Maintenance_Date'])
x_test = x_test.drop(
    columns=['Last_Maintenance_Date']
)


one_hot_encoder = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

maintenance_train = one_hot_encoder.fit_transform(
    x_train[['Maintenance_Type']]
)

maintenance_test = one_hot_encoder.transform(
    x_test[['Maintenance_Type']]
)

encoded_cols = one_hot_encoder.get_feature_names_out(
    ['Maintenance_Type']
)

print(encoded_cols)

maintenance_train_df = pd.DataFrame(
    maintenance_train,
    columns=encoded_cols,
    index=x_train.index
)

maintenance_test_df = pd.DataFrame(
    maintenance_test,
    columns=encoded_cols,
    index=x_test.index
)

x_train = pd.concat(
    [
        x_train.drop(columns=['Maintenance_Type']),
        maintenance_train_df
    ],
    axis=1
)

x_test = pd.concat(
    [
        x_test.drop(columns=['Maintenance_Type']),
        maintenance_test_df
    ],
    axis=1
)

print(x_train.head())
print(x_train.columns)

display(x_train.head())


print(x_train['Maintenance_Cost'].describe())

sns.histplot(x_train['Maintenance_Cost'], kde=True)
plt.show()

x_train['Maintenance_Cost'] = np.log1p(
    x_train['Maintenance_Cost']
)
x_test['Maintenance_Cost'] = np.log1p(
    x_test['Maintenance_Cost']
)

sns.histplot(x_train['Maintenance_Cost'], kde=True)
plt.show()


x_train=x_train.drop(columns=['Engine_Temperature'])
x_test=x_test.drop(columns=['Engine_Temperature'])


sns.histplot(x_train[ 'Tire_Pressure'], kde=True)
plt.show()

print(x_train['Tire_Pressure'].describe())

x_train['Tire_Pressure'] = np.log1p(
    x_train['Tire_Pressure']
)
x_test['Tire_Pressure'] = np.log1p(
    x_test['Tire_Pressure']
)

sns.histplot(x_train['Tire_Pressure'], kde=True)
plt.show()


sns.histplot(x_train['Fuel_Consumption'], kde=True)
plt.show()

print(x_train['Fuel_Consumption'].describe())

x_train['Fuel_Consumption'] = np.log1p(x_train['Fuel_Consumption'])
x_test['Fuel_Consumption'] = np.log1p(x_test['Fuel_Consumption'])


print(x_train['Battery_Status'].describe())

sns.histplot(x_train['Battery_Status'], kde=True)
plt.show()

x_train['Is_Battery_Elevated'] = (x_train['Battery_Status'] > 47.5).astype(int)
x_test['Is_Battery_Elevated'] = (x_test['Battery_Status'] > 47.5).astype(int)
x_train = x_train.drop(columns=['Battery_Status'])
x_test = x_test.drop(columns=['Battery_Status'])

display(x_train.head())

x_train['Is_Overloaded'] = (x_train['capacity_ratio'] > 1.0).astype(int)
x_test['Is_Overloaded'] = (x_test['capacity_ratio'] > 1.0).astype(int)


scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


svm=Pipeline([
    ('scaler',StandardScaler()),
    ('svm',SVC(kernel='rbf',C=1,probability=True,random_state=42))
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

voting.fit(x_train,y_train)

y_pred_voting=voting.predict(x_test)


cm=confusion_matrix(y_test,y_pred_voting)
plt.figure(figsize=(6,6))
sns.heatmap(
    cm,
    annot=True,
    cmap='Blues',
    fmt='d',
    cbar=False,
    xticklabels=['Normal', 'Minor', 'Major'],
    yticklabels=['Normal', 'Minor', 'Major']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.show()


print("Accuracy :", accuracy_score(y_test, y_pred_voting))

print("Precision:", precision_score(
    y_test, y_pred_voting, average='weighted'
))

print("Recall   :", recall_score(
    y_test, y_pred_voting, average='weighted'
))

print("F1-Score :", f1_score(
    y_test, y_pred_voting, average='weighted'
))


svm.fit(x_train, y_train)
rf.fit(x_train, y_train)
et.fit(x_train, y_train)

y_pred_svm = svm.predict(x_test)
y_pred_rf = rf.predict(x_test)
y_pred_et = et.predict(x_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Extra Trees Accuracy:", accuracy_score(y_test, y_pred_et))
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_voting))


y_train_pred=rf.predict(x_train)
y_test_pred=rf.predict(x_test)

print("Training Accuracy:",accuracy_score(y_train,y_train_pred))
print("Testing Accuracy:",accuracy_score(y_test,y_test_pred))


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train_scaled, y_train)

y_pred_knn = knn.predict(x_test_scaled)

print("KNN Accuracy:",
      accuracy_score(y_test, y_pred_knn))


logistic = LogisticRegression(max_iter=1000, random_state=42)

logistic.fit(x_train_scaled, y_train)

y_pred_logistic = logistic.predict(x_test_scaled)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_logistic))


models = {
    'SVM': y_pred_svm,
    'Random Forest': y_pred_rf,
    'Extra Trees': y_pred_et,
    'Voting Classifier': y_pred_voting,
    'KNN': y_pred_knn,
    'Logistic Regression': y_pred_logistic
}

results = []

for model_name, predictions in models.items():

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average='weighted'
    )

    recall = recall_score(
        y_test,
        predictions,
        average='weighted'
    )

    f1 = f1_score(
        y_test,
        predictions,
        average='weighted'
    )

    results.append([
        model_name,
        accuracy,
        precision,
        recall,
        f1
    ])

results_df = pd.DataFrame(
    results,
    columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
)

results_df


class_names = ['Normal', 'Minor', 'Major']

for model_name, predictions in models.items():

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.show()


results_df = results_df.sort_values(
    by='Accuracy',
    ascending=False
)

print("Model Comparison:")
display(results_df)

best_model = results_df.iloc[0]

print("\nBest Model:", best_model['Model'])
print("Accuracy:", round(best_model['Accuracy'], 4))
print("Precision:", round(best_model['Precision'], 4))
print("Recall:", round(best_model['Recall'], 4))


best_model_name = best_model['Model']
best_predictions = models[best_model_name]

print(f"Classification Report for Best Model: {best_model_name}\n")
print(classification_report(y_test, best_predictions))


print(x.columns.tolist())

print(pd.crosstab(
    df['Severity_Score'],
    df['Maintenance_Level_Code'],
    normalize='index'
))

print(pd.crosstab(
    df['Failure_History'],
    df['Maintenance_Level_Code'],
    normalize='index'
))

print(pd.crosstab(
    df['Anomalies_Detected'],
    df['Maintenance_Level_Code'],
    normalize='index'
))


from sklearn.model_selection import cross_val_score

rf_cv = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

scores = cross_val_score(
    rf_cv,
    x_train,
    y_train,
    cv=5,
    scoring='accuracy'
)

print("CV Scores:", scores)
print("Mean CV Accuracy:", scores.mean())


importance = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print(importance)


suspicious_features = [
    'Severity_Score',
    'Anomalies_Detected',
    'Brake_Condition',
    'Failure_History'
]

x_train_reduced = x_train.drop(columns=suspicious_features)
x_test_reduced = x_test.drop(columns=suspicious_features)

rf_reduced = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_reduced.fit(x_train_reduced, y_train)

y_pred_reduced = rf_reduced.predict(x_test_reduced)

print(
    "Random Forest Accuracy without strongest features:",
    accuracy_score(y_test, y_pred_reduced)
)


suspicious_features = [
    'Severity_Score',
    'Anomalies_Detected',
    'Brake_Condition',
    'Failure_History'
]

for feature in suspicious_features:

    x_train_test = x_train.drop(columns=[feature])
    x_test_test = x_test.drop(columns=[feature])

    rf_test = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    rf_test.fit(x_train_test, y_train)

    pred = rf_test.predict(x_test_test)

    print(
        feature,
        "removed → Accuracy:",
        accuracy_score(y_test, pred)
    )


correlation = df.corr(numeric_only=True)['Maintenance_Level_Code'] \
    .sort_values(ascending=False)

print(correlation)


print(y.value_counts())
print(y.value_counts(normalize=True))


print(pd.crosstab(
    df['Maintenance_Required'],
    df['Maintenance_Level_Code'],
    normalize='index'
))


print("Duplicate rows:", df.duplicated().sum())

print("Number of unique Vehicle IDs:", df['Vehicle_ID'].nunique())
print("Total rows:", len(df))


print(pd.crosstab(
    df['Maintenance_Type'],
    df['Maintenance_Level_Code'],
    normalize='index'
))


print(
    df.groupby('Maintenance_Level_Code')[
        ['Severity_Score',
         'Anomalies_Detected',
         'Failure_History']
    ].agg(['min', 'max', 'mean'])
)


print(
    pd.crosstab(
        df['Brake_Condition'],
        df['Maintenance_Level_Code'],
        normalize='index'
    )
)
