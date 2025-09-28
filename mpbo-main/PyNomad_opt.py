import numpy as np
import matplotlib.pyplot as plt
import PyNomad
from tqdm import tqdm

class PyNomadOptimizer:
    def __init__(self, search_space, response, prob=None):
        self.search_space = search_space
        self.obj = response
        self.prob = prob
        
    def bb(self, y):
        # print("y: ", y, "\n")
        # print("y: ", y.get_coord(0), "\n")
        # print("y.get_coord(0): ", int(y.get_coord(0)), "\n")
        
        # x = ch2xy[y[0]]
        # return response[y[0]]
        f = self.obj[int(y.get_coord(0))]
        # y.set_bb_output(0,f)
        y.setBBO(str(f).encode("UTF-8"))
        return 1 # 1: success
        
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
        
        index_bounds = [0, len(self.search_space) - 1]

        max_obj = self.obj.max()
        
        lb = [index_bounds[0]]
        ub = [index_bounds[1]]

        # for i in tqdm(range(repetitions)):
        for i in tqdm(range(repetitions)):
            # query = 0
            # print("follow_baseline: ", follow_baseline, "\n")
            if follow_baseline is not None:
                # print("Follow baseline \n")
                # while query < initial_points:
                for j in range(initial_points):
                    y0 = int(follow_baseline[i, j, 0].item()) # int(follow_baseline[i, query, 0].detach().numpy())
                    # print("y0 ", y0, "\n")
                    # self.results[i, j] = 1 - follow_baseline[i, j, 1] / max_obj
                    
                    # if self.prob == "NHP":
                    #     self.results[i, j] = 1+self.obj[y0]
                    # else:
                    self.results[i, j] = self.obj[y0] # 1 - follow_baseline[i, j, 1] / max_obj
                    # print("self.results[i, j]", self.results[i, j], "\n")
                    # follow_baseline[i, j, 1] # .item() # self.obj[y0]

                    # exploration_score[rep, query] = 1 - objective_func[best_query] / max_obj
                    # print("self.results[i, query]: ", self.results[i, j], "\n")

                    # query += 1

                # print("Done \n")

                params = ['BB_OUTPUT_TYPE OBJ','MAX_BB_EVAL 100', 'SEED 42', "DISPLAY_ALL_EVAL yes", "DISPLAY_STATS BBE OBJ", 'SOLUTION_FILE C:\\Users\\nicle\\Desktop\\MP-GPBO-Nic\\OG_MP_GPBO_code\\PyNomad_stats\\Ackley_results.txt'
                        , 'STATS_FILE C:\\Users\\nicle\\Desktop\\MP-GPBO-Nic\\OG_MP_GPBO_code\\PyNomad_stats\\Ackley_stats.txt']

                # self.results[i, 0] = self.obj[y0]
                # y0 = self.search_space[yi]
                # print("y0: ", y0, "\n")
                result = PyNomad.optimize(self.bb,[y0],lb,ub,params)
                # print("output: ", output, "\n")
                fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
                output = "\n".join(fmt)
                # print("\nNOMAD results \n" + output + " \n")
                
                with open("C:\\Users\\nicle\\Desktop\\MP-GPBO-Nic\\OG_MP_GPBO_code\\PyNomad_stats\\Ackley_stats.txt", "r") as file:
                    lines = file.readlines()[:iterations-initial_points]
                    second_column = [float(line.split()[1]) for line in lines]  # Extract the second column
                    self.results[i, j+1:len(second_column)+1] = second_column # 1 - second_column / max_obj
                    # second_column  # Save into the results array

            # if follow_baseline is not None and query < initial_points:
            #     optim_steps[rep, query, 0] = follow_baseline[rep, query, 0]
            #     optim_steps[rep, query, 1] = follow_baseline[rep, query, 1]

            else:
                y0 = inits[i]  # Start at the middle index

                # print("y0: ", y0, "\n")
            
                #params = ['BB_OUTPUT_TYPE OBJ','MAX_BB_EVAL 150', 'SEED 42', "DISPLAY_ALL_EVAL yes", "DISPLAY_STATS BBE OBJ", 'SOLUTION_FILE C:\\Users\\nicle\\Desktop\\MP-GPBO-Nic\\OG MP_GPBO code\\Results_PyNomad\\Ackley_results.txt'
                    #    , 'STATS_FILE C:\\Users\\nicle\\Desktop\\MP-GPBO-Nic\\OG MP_GPBO code\\Results_PyNomad\\Ackley_stats.txt']
                params = ['BB_OUTPUT_TYPE OBJ','MAX_BB_EVAL 100', 'SEED 42', "DISPLAY_ALL_EVAL yes", "DISPLAY_STATS BBE OBJ", 'SOLUTION_FILE C:\\Users\\nicle\\Desktop\\MP-GPBO-Nic\\OG_MP_GPBO_code\\PyNomad_stats\\Ackley_results.txt'
                        , 'STATS_FILE C:\\Users\\nicle\\Desktop\\MP-GPBO-Nic\\OG_MP_GPBO_code\\PyNomad_stats\\Ackley_stats.txt']

                # if self.prob == "NHP":
                #     self.results[i, 0] = 1+self.obj[y0]
                # else:
                self.results[i, 0] = self.obj[y0]
        
                result = PyNomad.optimize(self.bb,y0,lb,ub,params)
                # print("output: ", output, "\n")
                fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
                output = "\n".join(fmt)
                # print("\nNOMAD results \n" + output + " \n")
                
                with open("C:\\Users\\nicle\\Desktop\\MP-GPBO-Nic\\OG_MP_GPBO_code\\PyNomad_stats\\Ackley_stats.txt", "r") as file:
                    lines = file.readlines()[:iterations-1]
                    second_column = [float(line.split()[1]) for line in lines]  # Extract the second column
                    self.results[i, 1:len(second_column)+1] = second_column  # Save into the results array
        
        
        if save_file:
            np.savez(
                f"{file_name}.npz",
                repetitions=repetitions,
                iterations=iterations,
                strategy='PyNomad', 
                regret=self.results
            )
        return self.results

# class ObjectiveFunction_PyNomad: # Class for objective synthetic functions
#     def __init__(
#         self, benchmark, dim=2, size=None):
#         self.dim = dim
#         # self.negate = negate

#         if benchmark == "Ackley":
#             self.f = self.ackley_function

#             if size is None:
#                 self.size = 64
#             else:
#                 self.size = size
#             self.upper_bound = 32
#             self.lower_bound = -32
            
#         elif benchmark == "Hartmann":
#             try:
#                 assert dim == 6
#             except AssertionError:
#                 raise ValueError("Hartmann function is only defined for dim=6.")

#             self.f = self.hartmann_6d_function

#             self.size = 5
#             self.upper_bound = 1
#             self.lower_bound = 0

#         elif benchmark == "Michalewicz":
#             self.f = self.michalewicz_function

#             self.upper_bound = np.pi
#             self.lower_bound = 0
#             if self.dim == 2:
#                 self.size = 64

#             elif self.dim == 4:
#                 self.size = 10

#             else:
#                 raise ValueError("Choose dim=2 or dim=4 for Michalewicz function.")
#         else:
#             raise ValueError(
#                 "Choose a valid benchmark function in ['Ackley', 'Hartmann', 'Michalewicz']."
#             )

#     # Ackley Function
#     def ackley_function(self, x):
#         a = 20
#         b = 0.2
#         c = 2 * np.pi
#         d = len(x)
#         sum1 = np.sum(x**2)
#         sum2 = np.sum(np.cos(c * x))
#         return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e
    
#     def michalewicz_function(self, x):
#         d = len(x) # 10
#         return -np.sum(np.sin(x) * np.sin(np.arange(1, self.dim + 1) * x**2 / np.pi)**(2 * d))
    
#     def hartmann_6d_function(self, x):
#         """Computes the 6-dimensional Hartmann function"""
#         # Constants for the Hartmann 6D function
#         alpha = np.array([1.0, 1.2, 3.0, 3.2])
#         A = np.array([
#             [10, 3, 17, 3.50, 1.7, 8],
#             [0.05, 10, 17, 0.1, 8, 14],
#             [3, 3.5, 1.7, 10, 17, 8],
#             [17, 8, 0.05, 10, 0.1, 14]
#         ])
#         P = np.array([
#             [1312, 1696, 5569, 124, 8283, 5886],
#             [2329, 4135, 8307, 3736, 1004, 9991],
#             [2348, 1451, 3522, 2883, 3047, 6650],
#             [4047, 8828, 8732, 5743, 1091, 381]
#         ])*1e-4
        
        
#         x = np.asarray(x)  # Ensure the input is a numpy array
#         # x.reshape((1,6))
#         result = 0.0
#         # for i in range(4):  # Loop through the 4 components
#         #     sum_exp = 0.0
#         #     for j in range(6):  # Loop through the 6 dimensions
#         #         sum_exp += A[i][j] * (x[j] - P[i][j])**2
#         #     print("sum_exp", sum_exp, "\n")
#         #     print("alpha[i] ", alpha[i], "\n")
#         #     result -= alpha[i] * np.exp(-sum_exp)
#         #     print("result ", result, "\n")
#         # # return result

#         outer = 0
#         for ii in range(4):
#             inner = 0
#             for jj in range(6):
#                 xj = x[jj]
#                 Aij = A[ii, jj]
#                 Pij = P[ii, jj]
#                 inner = inner + Aij*(xj-Pij)**2

#             new = alpha[ii] * np.exp(-inner)
#             outer = outer + new

#         result = -outer
#         return result
        
#     # def bb_Ackley(self, y):
#     #     return obj.ackley_function(x)
    
#     # def initialize(self, ch2xy, initial_points=1, repetitions=30):
#     #     """
#     #     Generates initial points for the optimization

#     #     Parameters:
#     #         initial_points: the number of initial points to generate
#     #         repetitions: the number of repetitions for optimization
#     #     """
#     #     inits = np.random.randint(
#     #         0, ch2xy.shape[0], size=(repetitions, initial_points)
#     #     )
#     #     return inits
        
#     # Generate Input Space
#     def create_input_space(self):
#         X = np.linspace(self.lower_bound, self.upper_bound, self.size)
#         X = np.array([X for _ in range(self.dim)])
#         X = np.meshgrid(*X)
#         ch2xy = np.array([x.flatten() for x in X]).T
#         return ch2xy

#     # Generate Normalized True Response
#     def generate_true_response(self, input_space):
#         response = np.array([self.f(x) for x in input_space])
#         response = (response - response.min()) / (response.max() - response.min())
#         return response
    
import torch
import numpy as np

from botorch.test_functions import Ackley, Hartmann, Michalewicz


class ObjectiveFunction_PyNomad:
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


if __name__ == "__main__":

    obj_PyNomad = ObjectiveFunction_PyNomad("Ackley", dim=2)
    search_space_PyNomad = obj_PyNomad.create_input_space()
    response_PyNomad = obj_PyNomad.generate_true_response(search_space_PyNomad)

    opt_PyNomad = PyNomadOptimizer(search_space_PyNomad, response_PyNomad)
    regret_PyNomad = opt_PyNomad.train(repetitions=30, iterations=100) # , save_file=True, file_name="Ackley_PyNomad"


    # Initialize the Objective Function
    # obj = ObjectiveFunction_CMA_ES("Ackley", dim=2)
    # obj = ObjectiveFunction_min("Michalewicz", dim=2)
    # obj = ObjectiveFunction_min("Michalewicz", dim=4)
    # obj = ObjectiveFunction_min("Hartmann", dim=6)

    # Generate Input Space and True Responses
    # search_space = obj.create_input_space()
    # response = obj.generate_true_response(search_space)
    
    # opt_CMAES = CMA_ES_Optimizer(search_space, response)
    # regret_CMA_ES= opt_CMAES.train(repetitions=30, iterations=100) # , save_file=True, file_name="Ackley_PyNomad"

    plt.plot(regret_PyNomad.mean(axis=0), label="CMA-ES")
    # plt.fill_between(
    #     range(100),
    #     regret_CMA_ES.mean(0) - regret_CMA_ES.std(0) / np.sqrt(regret_CMA_ES.shape[0]),
    #     regret_CMA_ES.mean(0) + regret_CMA_ES.std(0) / np.sqrt(regret_CMA_ES.shape[0]),
    #     color="grey",
    #     alpha=0.2,
    # )
    
    plt.show()


# if __name__ == "__main__":
    # nb_reps = 30
    # results = np.zeros((nb_reps, 100))
    
    # # Initialize the Objective Function
    # obj = ObjectiveFunction_min("Ackley", dim=2)
    # # obj = ObjectiveFunction_min("Michalewicz", dim=2)
    # # obj = ObjectiveFunction_min("Michalewicz", dim=4)
    # # obj = ObjectiveFunction_min("Hartmann", dim=6)

    # # Generate Input Space and True Responses
    # ch2xy = obj.create_input_space()
    # response = obj.generate_true_response(ch2xy)
    
    # print("Input space shape:", ch2xy.shape)
    # print("Response shape:", response.shape)
    
    # def bb(y):
    #     # print("y: ", y, "\n")
    #     # print("y: ", y.get_coord(0), "\n")
    #     # print("y.get_coord(0): ", int(y.get_coord(0)), "\n")
        
    #     # x = ch2xy[y[0]]
    #     # return response[y[0]]
    #     f = response[int(y.get_coord(0))]
    #     # y.set_bb_output(0,f)
    #     y.setBBO(str(f).encode("UTF-8"))
    #     return 1 # 1: success

    # seed = 42
    # np.random.seed(seed)
    
    # inits = obj.initialize(ch2xy, initial_points=1, repetitions=nb_reps)

    # # CMA-ES will optimize indices within the range [0, len(ch2xy)-1]
    # index_bounds = [0, len(ch2xy) - 1]
    
    # lb = [index_bounds[0]]
    # ub = [index_bounds[1]]
    
    # for i in range(nb_reps):
    #     y0 = inits[i]  # Start at the middle index
    
    #     params = ['BB_OUTPUT_TYPE OBJ','MAX_BB_EVAL 150', 'SEED 42', "DISPLAY_ALL_EVAL yes", "DISPLAY_STATS BBE OBJ", 'SOLUTION_FILE C:\\Users\\nicle\\anaconda3\\envs\\test2\\Results\\Ackley_results.txt'
    #             , 'STATS_FILE C:\\Users\\nicle\\anaconda3\\envs\\test2\Results\\Ackley_stats.txt']
    
    #     result = PyNomad.optimize(bb,y0,lb,ub,params)
    #     # print("output: ", output, "\n")
    #     fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
    #     output = "\n".join(fmt)
    #     print("\nNOMAD results \n" + output + " \n")
        
    #     with open("C:\\Users\\nicle\\anaconda3\\envs\\test2\Results\\Ackley_stats.txt", "r") as file:
    #         lines = file.readlines()[:100]
    #         second_column = [float(line.split()[1]) for line in lines]  # Extract the second column
    #         results[i, :len(second_column)] = second_column  # Save into the results array

