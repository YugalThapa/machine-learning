import numpy as np
import matplotlib.pyplot as plt

# Least Square method
def least_squares(X, Y):
    
    n = len(X)
    m = (n*np.sum(X*Y) - np.sum(X)*np.sum(Y)) / (n*np.sum(np.square(X)) - np.square(np.sum(X)))
    c = (np.sum(Y) - m*np.sum(X)) / n
    return m, c

# Gradient Descent Algorithm
def gradient_descent(X, Y, learning_rate=0.0001, iterations=1000):

    m, b = 1, 0
    n = len(X)

    for _ in range(iterations):
        grad_m = (-2/n) * np.sum(X * (Y - (m*X + b)))
        grad_b = (-2/n) * np.sum(Y - (m*X + b))
        m = m - learning_rate * grad_m
        b = b - learning_rate * grad_b

    return m, b

# Mean Square Error
def mse(X, Y, m, c):
   
    predictions = m * X + c
    return np.mean((Y - predictions) ** 2)

# Scatter plot for visualization
def plot(X, Y, m_ls, c_ls, m_gd, c_gd):
    
    x_line = np.linspace(0, 100, 100)

    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', label='Data Points', alpha=0.6)
    plt.plot(x_line, m_ls*x_line + c_ls, color='red',   label='Least Squares')
    plt.plot(x_line, m_gd*x_line + c_gd, color='green', label='Gradient Descent', linestyle='--')
    plt.title('Linear Regression from Scratch')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    np.random.seed(42)
    X = np.random.rand(50, 1) * 100
    Y = 3.5 * X + np.random.randn(50, 1) * 20

    x_input = 44

    # Least squares
    m_ls, c_ls = least_squares(X, Y)
    print("--- Least Squares ---")
    print(f"  Slope: {m_ls:.4f}  |  Intercept: {c_ls:.4f}")
    print(f"  MSE:   {mse(X, Y, m_ls, c_ls):.4f}")
    print(f"  Prediction for X={x_input}: {(m_ls*x_input + c_ls).item():.4f}")

    print()

    # Gradient descent
    m_gd, c_gd = gradient_descent(X, Y)
    print("--- Gradient Descent ---")
    print(f"  Slope: {m_gd:.4f}  |  Intercept: {c_gd:.4f}")
    print(f"  MSE:   {mse(X, Y, m_gd, c_gd):.4f}")
    print(f"  Prediction for X={x_input}: {(m_gd*x_input + c_gd).item():.4f}")

    plot(X, Y, m_ls, c_ls, m_gd, c_gd)


if __name__ == "__main__":
    main()