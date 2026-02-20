import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset.csv")

# Features and Target
X = data[["Hours_Studied", "Attendance", "Previous_Score"]]
y = data["Final_Score"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Predict custom student
hours = 5
attendance = 85
previous_score = 75

prediction = model.predict([[hours, attendance, previous_score]])

print("\nPredicted Final Score for:")
print(f"Hours Studied: {hours}")
print(f"Attendance: {attendance}")
print(f"Previous Score: {previous_score}")
print("Predicted Final Score:", prediction[0])

# Simple visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Scores")
plt.show()
