import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# load data
df = pd.read_csv(r'D:\MachineLearning\Phase1-MlFoundations\Day3–5\student_scores.csv')
# print(df.head())
# print(df.describe())

# independent -> study hours, dependent -> marks scored
X = df[['study_hours']]
Y = df[['exam_score']]

# split data into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# model creation
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print(f"slope: {model.coef_[0][0]:.4f}")
print(f"intercept: {model.intercept_[0]:.4f}")

# evaluation
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)

print(f"MSE: {mse:.4F}")
print(f"r2: {r2:.4F}")
print(f"MAE: {mae:.4F}")

# predicting new value
study_hrs = pd.DataFrame([[2.25],[4],[8]], columns=['study_hours'])
predicted_score = model.predict(study_hrs)
print(f"Score predicted: {predicted_score}")

# plot visualization
x_line = np.linspace(0, 10, 100).reshape(-1, 1)

plt.figure(figsize=(8, 5))
plt.scatter(X_train, Y_train, color='blue',  alpha=0.5, label='Train data')
plt.scatter(X_test,  Y_test,  color='green', alpha=0.7, label='Test data')
plt.plot(x_line, model.predict(x_line), color='red', linewidth=2, label='Regression line')
plt.scatter(study_hrs, predicted_score, color='orange', s=100, zorder=5, label='New prediction')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Linear Regression — Sklearn')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()