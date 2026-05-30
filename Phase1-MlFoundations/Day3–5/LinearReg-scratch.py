import numpy as np
import matplotlib.pyplot as plt

# Function to predict Y
def predict_y(X,Y,x_input):
    n = len(X)
    m = ( n*np.sum(X*Y) - (np.sum(X)*np.sum(Y)) ) / ( n*np.sum(np.square(X)) - np.square(np.sum(X)) )
    c = (( np.sum(Y) - m*np.sum(X) ) / n)

    print(f"Slope (m): {m:.4f}")      # should be close to 3.5
    print(f"Intercept (c): {c:.4f}")  # should be close to 0

    return m*x_input + c

def scatter_plot(X,Y):
    plt.figure(figsize=(8,6)) 
    plt.scatter(X, Y, color='blue', label='Data Points') 
    plt.title('Linear Regression on Random Dataset')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # initializing a dataset
    np.random.seed(42)
    X = np.random.rand(50, 1) * 100  
    Y = 3.5 * X + np.random.randn(50, 1) * 20

    # scatter_plot(X,Y)
    x_input = 56
    y_predicted= predict_y(X,Y,x_input)
    print(f"Predicted y of x={x_input}: {y_predicted}")

if __name__ == "__main__":
    main()