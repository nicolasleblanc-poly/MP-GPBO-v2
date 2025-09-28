import PyNomad
import numpy as np
import matplotlib.pyplot as plt


# Class for objective synthetic functions
class ObjectiveFunction_min:
    def __init__(
        self, benchmark, dim=2, size=None):
        self.dim = dim
        self.f = self.hartmann_6d_function

        try:
            assert dim == 6
        except AssertionError:
            raise ValueError("Hartmann function is only defined for dim=6.")

        self.f = self.hartmann_6d_function

        self.size = 5
        self.upper_bound = 1
        self.lower_bound = 0


    def hartmann_6d_function(self, x):
        """Computes the 6-dimensional Hartmann function"""
        # Constants for the Hartmann 6D function
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([
            [10, 3, 17, 3.50, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ])
        P = np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ])*1e-4
    
    
        x = np.asarray(x)  # Ensure the input is a numpy array
        # x.reshape((1,6))
        result = 0.0
        # for i in range(4):  # Loop through the 4 components
        #     sum_exp = 0.0
        #     for j in range(6):  # Loop through the 6 dimensions
        #         sum_exp += A[i][j] * (x[j] - P[i][j])**2
        #     print("sum_exp", sum_exp, "\n")
        #     print("alpha[i] ", alpha[i], "\n")
        #     result -= alpha[i] * np.exp(-sum_exp)
        #     print("result ", result, "\n")
        # # return result

        outer = 0
        for ii in range(4):
            inner = 0
            for jj in range(6):
                xj = x[jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]
                inner = inner + Aij*(xj-Pij)**2

            new = alpha[ii] * np.exp(-inner)
            outer = outer + new

        result = -outer
        return result

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


x0 = np.random.rand(6)
lb = 
up = 

