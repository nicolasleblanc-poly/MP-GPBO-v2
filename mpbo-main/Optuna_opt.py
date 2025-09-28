import numpy as np
import matplotlib.pyplot as plt
import optuna
# from smac.initial_design import SingleConfigInitialDesign
from ConfigSpace import ConfigurationSpace
from ConfigSpace import Configuration
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.ERROR)

class OptunaOptimizer:
    def __init__(self, search_space, response, prob=None):
        self.search_space = search_space
        self.obj = response
        self.prob = prob
        
        self.step = 1 #(obj.upper_bound - obj.lower_bound) / (obj.size - 1)
        
    # Objective function
    def objective(self, trial):
        index = trial.suggest_float(f"x", 0, len(self.search_space)-1, step=self.step) # for i in range(obj.dim)
        # print(index)
        # x = search_space[int(index)]
        # print("x ", x, "\n")
        # val = obj.f(torch.tensor(x))
        val = self.obj[int(index)]
        return val #.item()
        
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

    def train(self, initial_points=1, repetitions=30, iterations=100, follow_baseline=None, save_file=False, file_name=None):
        self.results = np.zeros((repetitions, iterations))
        inits = self.initialize(initial_points=initial_points, repetitions=repetitions)
        
        cs = ConfigurationSpace({"x": (0, len(self.search_space)-1)})
        
        for i in tqdm(range(repetitions)):
            study = optuna.create_study(direction="minimize")
            
            if follow_baseline is not None:
                for j in range(initial_points):
                    starting_index = int(follow_baseline[i, j, 0].item())
                    study.enqueue_trial({"x": starting_index})
                    self.results[i, j] = self.obj[starting_index]
            else:
                starting_index = inits[i]
                study.enqueue_trial({"x": starting_index})
                self.results[i, 0] = self.obj[starting_index]
                
            study.optimize(self.objective, n_trials=100)
            
            # Extract optimization history
            history = study.trials_dataframe()
            history_indices = history['params_x'].apply(lambda x: int(round(x))).values
            history_values = history['value'].values
            history_coordinates = [self.search_space[idx] for idx in history_indices]
            
            # Store results
            trial_count = min(iterations, len(history_values))
            self.results[i, :trial_count] = history_values[-trial_count:]
        
        if save_file:
            np.savez(
                f"{file_name}.npz",
                repetitions=repetitions,
                iterations=iterations,
                strategy='Optuna', 
                regret=self.results
            )
        return self.results
    
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

    opt_Optuna = OptunaOptimizer(search_space_PyNomad, response_PyNomad)
    regret_Optuna = opt_Optuna.train(repetitions=30, iterations=100) # , save_file=True, file_name="Ackley_PyNomad"

    plt.plot(regret_Optuna.mean(axis=0), label="Optuna")
    
    plt.plot(np.minimum.accumulate(regret_Optuna.mean(axis=0)), 'r-', label='Best so far')
    
    plt.show()
