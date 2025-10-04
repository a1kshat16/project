# ------------------------------------------
# Comprehensive Water Quality Analysis
# ------------------------------------------

# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
data = pd.read_csv("waterpotability.csv")

print(" Dataset Loaded Successfully!")
print(data.head())

# ------------------------------------------
# Data Cleaning
# ------------------------------------------
print("\n Checking for Missing Values:")
print(data.isnull().sum())

# Fill missing values using median
data = data.fillna(data.median())

# Verify again
print("\n Missing Values After Cleaning:")
print(data.isnull().sum())

# ------------------------------------------
# Data Visualization
# ------------------------------------------
print("\n Generating Basic Visualizations...")

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Distribution of Potability
sns.countplot(x='Potability', data=data, palette='Set2')
plt.title("Potable vs Non-Potable Water Count")
plt.show()

# Boxplot for pH
sns.boxplot(x='Potability', y='ph', data=data)
plt.title("pH Distribution by Potability")
plt.show()

# ------------------------------------------
# Machine Learning Model
# ------------------------------------------
print("\n Building Machine Learning Model...")

X = data.drop('Potability', axis=1)
y = data['Potability']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {acc:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# Save results for Tableau dashboard
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results.to_csv('predicted_results.csv', index=False)
print("\n Results saved for Tableau dashboard: predicted_results.csv")

print("\n Project Completed Successfully - Akshat Kumar")
