
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data: Area (sq ft) vs. Price ($)
X = np.array([500, 700, 800, 1000, 1200]).reshape(-1, 1)
y = np.array([150000, 200000, 230000, 275000, 320000])

# Create and Train Model
model = LinearRegression()
model.fit(X, y)

# Make a Prediction for 900 sq ft
predicted_price = model.predict([[900]])
print(f"Predicted Price for 900 sq ft: ${predicted_price[0]:.2f}")

# Plot Data and Regression Line
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Prediction Line")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.legend()
plt.show()
