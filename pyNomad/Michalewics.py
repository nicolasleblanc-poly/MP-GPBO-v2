import PyNomad
import numpy as np
import matplotlib.pyplot as plt

# Class for objective synthetic functions
class ObjectiveFunction_min:
    def __init__(
        self, benchmark, dim=2, size=None):
        self.dim = dim
        self.f = self.michalewicz_function

        self.upper_bound = np.pi
        self.lower_bound = 0
        if self.dim == 2:
            self.size = 64

        elif self.dim == 4:
            self.size = 10

        else:
            raise ValueError("Choose dim=2 or dim=4 for Michalewicz function.")


    def michalewicz_function(self, x):
        d = len(x) # 10
        return -np.sum(np.sin(x) * np.sin(np.arange(1, self.dim + 1) * x**2 / np.pi)**(2 * d))

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

