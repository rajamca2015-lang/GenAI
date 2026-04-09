import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# Dataset
data = {
    "TV": [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 144.1, 111.6],
    "Radio": [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 16.0, 12.6],
    "Newspaper": [69.2, 45.1, 69.3, 58.5, 58.4, 75.0, 23.5, 11.6, 40.3, 37.9],
    "Sales": [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 15.6, 12.2]
}
df = pd.DataFrame(data)

# Train regression model
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]
model = LinearRegression().fit(X, y)

# --- Scatter plots with regression lines ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feature in enumerate(["TV", "Radio", "Newspaper"]):
    sns.scatterplot(x=df[feature], y=df["Sales"], ax=axes[idx])
    # Fit line for each feature individually
    coef = np.polyfit(df[feature], df["Sales"], 1)
    poly1d_fn = np.poly1d(coef)
    axes[idx].plot(df[feature], poly1d_fn(df[feature]), color='red')
    axes[idx].set_title(f"{feature} Budget vs Sales")

plt.show()

# Predictions
y_pred = model.predict(X)

# --- Diagnostics ---
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("R² Score:", r2_score(y, y_pred))
print("Mean Squared Error:", mean_squared_error(y, y_pred))

# Residuals
residuals = y - y_pred

# --- Residual Plot ---
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# --- Actual vs Predicted Plot ---
plt.scatter(y, y_pred)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red')  # perfect fit line
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()



