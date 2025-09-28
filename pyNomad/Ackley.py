import PyNomad
import numpy as np
import matplotlib.pyplot as plt



# Class for objective synthetic functions
class ObjectiveFunction_min:
    def __init__(
        self, benchmark, dim=2, size=None):
        self.dim = dim
        self.f = self.ackley_function

        if size is None:
            self.size = 64
        else:
            self.size = size
        self.upper_bound = 32
        self.lower_bound = -32

    # Ackley Function
    def ackley_function(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e

    # Generate Input Space
    def create_input_space(self):
        X = np.linspace(self.lower_bound, self.upper_bound, self.size)
        X = np.array([X for _ in range(self.dim)])
        X = np.meshgrid(*X)
        ch2xy = np.array([x.flatten() for x in X]).T
        return ch2xy

    # Generate Normalized True Response
    def generate_true_response(self, input_space):
        response = np.array([self.f(x) for x in input_space])
        response = (response - response.min()) / (response.max() - response.min())
        return response


obj = ObjectiveFunction_min("Auckley", dim=2)

# Generate Input Space and True Responses
ch2xy = obj.create_input_space()
response = obj.generate_true_response(ch2xy)

print("Input space shape:", ch2xy.shape)
print("Response shape:", response.shape)

# Initialize the CMA-ES Optimization
seed = 42
np.random.seed(seed)

# CMA-ES will optimize indices within the range [0, len(ch2xy)-1]
index_bounds = [0, len(ch2xy) - 1]
x0 = [len(ch2xy) // 2]  # Start at the middle index

