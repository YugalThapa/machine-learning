import numpy as np
import matplotlib.pyplot as plt
from   sklearn.linear_model import LinearRegression

# Generating random dataset
np.random.seed(42)
X = np.random.rand(50, 1) * 100  
Y = 3.5 * X + np.random.randn(50, 1) * 20

# creating and training linear regression model
model = LinearRegression()
model.fit(X,Y)

y_predict = model.predict(X)

# visualize regression line
plt.figure(figsize=(8,6)) 
plt.scatter(X, Y, color='blue', label='Data Points') 
plt.plot(X, y_predict, color='red', linewidth=2, label='Regression Line') 
plt.title('Linear Regression on Random Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

print("Slope (Coefficient):", model.coef_[0][0])
print("Intercept:", model.intercept_[0])