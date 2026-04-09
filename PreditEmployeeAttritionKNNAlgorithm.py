import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load dataset (Example)
data = {
    "Age": [29, 35, 40, 28, 45, 25, 50, 30, 37, 26],
    "JobRole": ["Sales Executive", "Research Scientist", "Laboratory Technician",
                "Sales Executive", "Manager", "Research Scientist", "Manager",
                "Sales Executive", "Laboratory Technician", "Research Scientist"],
    "MonthlyIncome": [4800, 6000, 3400, 4300, 11000, 3500, 12000, 5000, 3100, 4500],
    "JobSatisfaction": [3, 4, 2, 3, 4, 1, 4, 2, 2, 3],
    "YearsAtCompany": [4, 8, 6, 3, 15, 2, 20, 5, 9, 2],
    "Attrition": [1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
}
df = pd.DataFrame(data)

# 2. Encode categorical feature (JobRole)
encoder = LabelEncoder()
df["JobRole"] = encoder.fit_transform(df["JobRole"])

# 3. Define features and target
X = df[["Age", "JobRole", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany"]]
y = df["Attrition"]

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 6. Build KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 7. Predictions
y_pred = knn.predict(X_test)

# 8. Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example prediction: new employee
new_employee = pd.DataFrame(
    [[32, encoder.transform(["Sales Executive"])[0], 5200, 3, 4]],
    columns=["Age", "JobRole", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany"]
)

# Scale with consistent feature names
new_employee_scaled = scaler.transform(new_employee)

# Predict
print("Predicted Attrition (1=Yes, 0=No):", knn.predict(new_employee_scaled)[0])


