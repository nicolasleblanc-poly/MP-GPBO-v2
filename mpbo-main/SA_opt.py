import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings 
warnings.filterwarnings("ignore")
from abc import ABCMeta, abstractmethod
import types
import warnings
import pandas as pd
from sko.SA import SAFast
import torch
from botorch.test_functions import Ackley, Hartmann, Michalewicz

class ObjectiveFunction_SA:
    def __init__(
        self, benchmark, dim=2, size=None, negate=False, device=torch.device("cpu")
    ):
        self.dim = dim
        self.negate = negate
        self.device = device

        if benchmark == "Ackley":
            self.f = Ackley(dim=self.dim, negate=self.negate)

            if size is None:
                self.size = 64
            else:
                self.size = size
            self.upper_bound = 32
            self.lower_bound = -32

        elif benchmark == "Hartmann":
            try:
                assert dim == 6
            except AssertionError:
                raise ValueError("Hartmann function is only defined for dim=6.")

            self.f = Hartmann(dim=self.dim, negate=self.negate)

            self.size = 5
            self.upper_bound = 1
            self.lower_bound = 0

        elif benchmark == "Michalewicz":
            self.f = Michalewicz(dim=self.dim, negate=self.negate)

            self.upper_bound = np.pi
            self.lower_bound = 0
            if self.dim == 2:
                self.size = 64

            elif self.dim == 4:
                self.size = 10

            else:
                raise ValueError("Choose dim=2 or dim=4 for Michalewicz function.")
        else:
            raise ValueError(
                "Choose a valid benchmark function in ['Ackley', 'Hartmann', 'Michalewicz']."
            )

    def create_input_space(self):
        X = np.linspace(self.lower_bound, self.upper_bound, self.size)
        X = np.array([X for _ in range(self.dim)])
        X = np.meshgrid(*X)
        ch2xy = np.array([x.flatten() for x in X]).T
        # ch2xy = torch.from_numpy(ch2xy).float().to(self.device)

        return ch2xy

    def generate_true_response(self, input_space):
        response = np.array([self.f(torch.tensor([*x])) for x in input_space])
        response = (response - response.min()) / (response.max() - response.min())
        # response = torch.from_numpy(response).float().to(self.device)
        return response

# Wrapper for the Objective Function for CMA-ES
class SA_IndexObjective:
    def __init__(self, search_space, response):
        self.search_space = search_space
        self.response = response

    def __call__(self, index):
        # Ensure the index is within bounds and is an integer
        idx = int(np.clip(np.floor(index[0]), 0, len(self.search_space) - 1))
        return self.response[idx]

class SA_Optimizer:
    def __init__(self, search_space, response, prob=None):
        self.search_space = search_space
        self.response = response
        self.prob = prob
        
    def initialize(self, initial_points=1, repetitions=30):
        """
        Generates initial points for the optimization

        Parameters:
            initial_points: the number of initial points to generate
            repetitions: the number of repetitions for optimization
        """
        inits = np.random.randint(
            0, self.search_space.shape[0], size=(repetitions, initial_points)
        )
        return inits
        
    def train(self, 
              initial_points=1,
              repetitions=30,
              iterations=100,
              follow_baseline=None,
              save_file=False,
              file_name=None,):
        self.results = np.zeros((repetitions, iterations))
        # ch2xy = self.obj.create_input_space()
        # response = self.obj.generate_true_response(ch2xy)
        inits = self.initialize(initial_points=initial_points, repetitions=repetitions)
        
        # results = np.zeros((repetitions, iterations))
        
        index_bounds = [0, len(self.search_space) - 1]
        
        # Objective Function for CMA-ES
        index_objective = SA_IndexObjective(self.search_space, self.response)
        
        for i in tqdm(range(repetitions)):
            x0 = inits[i]  # Start at the middle index

            if follow_baseline is not None:
                for j in range(initial_points):
                    x0 = int(follow_baseline[i,j,0].item())
                    # print("x0 ", x0, "\n")
                    
                    # if self.prob == "NHP":
                    #     self.results[i, j] = 1+self.response[x0]
                    # else:
                    self.results[i, j] = self.response[x0]
                    # print("self.response[x0] ", self.response[x0], "\n")
                    # print("self.results[i, j] ", self.results[i, j], "\n")
                    # print("i,j ", i, j, "\n")

                # x0 = int(follow_baseline[i,0,0].item())
                # print("x0 ", x0, "\n")
                # self.results[i, 0] = self.response[x0]
                # print("self.response[x0] ", self.response[x0], "\n")
                # print("self.results[i, 0] ", self.results[i, 0], "\n")
                # print("i,j ", i, 0, "\n")


                sa = SAFast(func=index_objective, x0=[x0], T_max=100, T_min=1e-14, L=300, max_stay_counter=150, lb=np.array([0]), 
                            ub=np.array([len(self.response) - 1]), max_nb_it=iterations-initial_points)

                # from sko.SA import SAFast

                # sa_fast = SAFast(func=demo_func, x0=[1, 1, 1], T_max=1, T_min=1e-9, q=0.99, L=300, max_stay_counter=150,
                #                 lb=[-1, 1, -1], ub=[2, 3, 4], max_iter=1000)
                # sa_fast.run()

                best_x, best_y = sa.run()

                # results[i, 0] = self.response[x0]
                # print("self.response[x0] ", self.response[x0], "\n")
                self.results[i, 1:] = pd.DataFrame(sa.best_y_history).cummin(axis=0)[:iterations-initial_points].values.flatten()
                # results[i, j+1:] = pd.DataFrame(sa.best_y_history).cummin(axis=0)[:iterations-initial_points].values.flatten()
                # print("self.results[i, 0] ", self.results[i, 0], "\n")
                # print("results[i,:] ", results[i,:], "\n")

            else:

                # sa = SAFast(func=index_objective, x0=[np.random.uniform(0, len(self.response))], T_max=100, T_min=1e-14, L=300, max_stay_counter=150, lb=np.array([0]), ub=np.array([len(self.response) - 1]), max_nb_it=iterations)

                sa = SAFast(func=index_objective, x0=[x0], T_max=100, T_min=1e-14, L=300, max_stay_counter=150, lb=np.array([0]), 
                            ub=np.array([len(self.response) - 1]), max_nb_it=iterations-initial_points)

                best_x, best_y = sa.run()
        
                # CMA-ES configuration to force 150 iterations
                # es = cma.CMAEvolutionStrategy(
                    # x0, 10, {
                    #     'bounds': index_bounds,
                    #     'maxiter': iterations,           # Force 150 iterations
                    #     'tolflatfitness': 0,      # Disable stopping due to flat fitness
                    #     'tolfun': 0,              # Disable stopping based on fitness improvement
                    #     'tolx': 0,                # Disable stopping based on step size
                    #     'tolfunhist': 0,          # Disable stopping based on fitness history
                    #     # 'verbose': 3,             # Enable detailed output
                    #     'seed': 42                # Fix the random seed for reproducibility
                    # })

                # fitness_values = []
                
                # # results[i, iteration] = self.response[x0]
            
                # if self.prob == "NHP":
                #     self.results[i, 0] = 1+self.response[x0]
                #     # print("self.response[x0] ", self.response[x0], "\n")
                #     self.results[i, j+1:] = 1+pd.DataFrame(sa.best_y_history).cummin(axis=0)[:iterations-1].values.flatten()
                # else:
                self.results[i, 0] = self.response[x0]
                # print("self.response[x0] ", self.response[x0], "\n")
                self.results[i, 1:] = pd.DataFrame(sa.best_y_history).cummin(axis=0)[:iterations-1].values.flatten()
                    # print("results[i, 0] ", results[i, 0], "\n")

            
            # iteration = 1
            # while iteration < iterations:
            #     solutions = es.ask()  # Generate candidate solutions (indices)
            #     fitness = [index_objective(x) for x in solutions]  # Evaluate responses
            #     es.tell(solutions, fitness)  # Update CMA-ES state
            #     # es.disp()

            #     # Save progress
            #     # fitness_values.append(es.result.fbest)
            #     results[i, iteration] = es.result.fbest
            #     iteration += 1

            # Final Results
            # best_index = int(np.floor(es.result.xbest[0]))
            # best_point = self.search_space[best_index]
            # best_response = self.response[best_index]
        # print("self.results[:,0] ", self.results[:,0], "\n")
        return self.results

if __name__ == "__main__":
    # Initialize the Objective Function
    obj = ObjectiveFunction_SA("Ackley", dim=2)
    # obj = ObjectiveFunction_min("Michalewicz", dim=2)
    # obj = ObjectiveFunction_min("Michalewicz", dim=4)
    # obj = ObjectiveFunction_min("Hartmann", dim=6)

    # SA_fast is the default
    # SA_custom = SAFast

    # Generate Input Space and True Responses
    search_space = obj.create_input_space()
    response = obj.generate_true_response(search_space)
    
    opt_SA = SA_Optimizer(search_space, response)
    regret_SA = opt_SA.train(repetitions=30, iterations=100) # , save_file=True, file_name="Ackley_PyNomad"

    plt.plot(regret_SA.mean(axis=0), label="SA")
    # plt.fill_between(
    #     range(100),
    #     regret_CMA_ES.mean(0) - regret_CMA_ES.std(0) / np.sqrt(regret_CMA_ES.shape[0]),
    #     regret_CMA_ES.mean(0) + regret_CMA_ES.std(0) / np.sqrt(regret_CMA_ES.shape[0]),
    #     color="grey",
    #     alpha=0.2,
    # )
    
    plt.show()
    