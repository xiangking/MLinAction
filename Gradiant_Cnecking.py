import numpy as np

def sigmoid(z):
    return 1./(1+np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
def check_gradient(f, x0, epsilon):
    return (f(x0+epsilon) - f(x0-epsilon))/2/epsilon

if __name__ == '__main__':
    x0 = np.array([1, 2, 3])
    epsilon = 1e-4
    print(sigmoid_prime(x0))
            # [ 0.19661193  0.10499359  0.04517666]
    print(check_gradient(sigmoid, x0, epsilon))
            # [ 0.19661193  0.10499359  0.04517666]