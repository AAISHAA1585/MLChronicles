 #Predict if a student will pass an exam based on study hours. by using Logistic Regression (for Classification)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Sample Data: Study Hours vs. Pass/Fail (1 = Pass, 0 = Fail)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Train Model
model = LogisticRegression()
model.fit(X, y)

# Predict for 4.5 hours
prediction = model.predict([[4.5]])
print(f"Will a student pass if they study 4.5 hours? {'Yes' if prediction[0] == 1 else 'No'}")

# Visualize
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict_proba(X)[:, 1], color="red", label="Prediction Curve")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.legend()
plt.show()
