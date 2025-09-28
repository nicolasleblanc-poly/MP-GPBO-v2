import numpy as np
import matplotlib.pyplot as plt
from smac import Scenario, HyperparameterOptimizationFacade, RunHistory
# from smac.initial_design import SingleConfigInitialDesign
from ConfigSpace import ConfigurationSpace
from ConfigSpace import Configuration
from tqdm import tqdm

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.runhistory.dataclasses import TrialValue, TrialInfo
# from smac import InitialDesign

import logging
logging.getLogger("smac").setLevel(logging.WARNING)

class SMACOptimizer:
    def __init__(self, search_space, response, prob=None):
        self.search_space = search_space
        self.obj = response
        self.prob = prob
        
        # Lists to store optimization history
        self.history_indices = []
        self.history_values = []
        self.history_coordinates = []
        
    # Objective function
    def smac_objective(self, config, seed=42):
        # print("Config:", config)
        index = config[f"x"]
        val = self.obj[int(index)]
        
        # Lists to store optimization history
        self.history_indices.append(index)
        self.history_values.append(val)
        self.history_coordinates.append(self.search_space[index])
        
        val = self.obj[int(index)]
        return val
        
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
            # Create initial configurations
            if follow_baseline is not None:
                initial_configs = []
                # costs = []
                for j in range(initial_points):
                    value = int(follow_baseline[i, j, 0].item())
                    config = Configuration(cs, values={"x": value})
                    initial_configs.append(config)
                    self.results[i, j] = self.obj[value]
            else:
                value = inits[i]
                config = Configuration(cs, values={"x": value})
                initial_configs = [config]
                self.results[i, 0] = self.obj[value]
            
            # Create scenario with custom initial design
            scenario = Scenario(cs, n_trials=iterations)
            
            costs = [self.smac_objective(config) for config in initial_configs]
            
            smac = HPOFacade(
                scenario,
                self.smac_objective,
                # initial_design=CustomInitialDesign(scenario, initial_configs),
                overwrite=True,
                initial_design=HyperparameterOptimizationFacade.get_initial_design(scenario,
                            n_configs=0, additional_configs=initial_configs,
                                                                                   )
            )
            
            trial_infos = [TrialInfo(config=c, seed=42) for c in initial_configs]
            trial_values = [TrialValue(cost=c) for c in costs]
            
            # Warmstart SMAC with the trial information and values
            for info, value in zip(trial_infos, trial_values):
                smac.tell(info, value)
            
            # incumbent = 
            smac.optimize()
            
            # Store results
            trial_count = min(iterations, len(self.history_values))
            self.results[i, :trial_count] = self.history_values[-trial_count:]
        
        if save_file:
            np.savez(
                f"{file_name}.npz",
                repetitions=repetitions,
                iterations=iterations,
                strategy='SMAC', 
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

    opt_SMAC = SMACOptimizer(search_space_PyNomad, response_PyNomad)
    regret_SMAC = opt_SMAC.train(repetitions=30, iterations=100) # , save_file=True, file_name="Ackley_PyNomad"

    plt.plot(regret_SMAC.mean(axis=0), label="SMAC")
    
    plt.plot(np.minimum.accumulate(regret_SMAC.mean(axis=0)), 'r-', label='Best so far')
    
    # plt.fill_between(
    #     range(100),
    #     regret_CMA_ES.mean(0) - regret_CMA_ES.std(0) / np.sqrt(regret_CMA_ES.shape[0]),
    #     regret_CMA_ES.mean(0) + regret_CMA_ES.std(0) / np.sqrt(regret_CMA_ES.shape[0]),
    #     color="grey",
    #     alpha=0.2,
    # )
    
    plt.show()
