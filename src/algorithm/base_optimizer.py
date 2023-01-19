import numpy as np
import pandas as pd

from src.loss.fairness import get_fairness
from src.loss.utility import get_utility
from src.utils.model import get_delta


class BaseAlgorithm(object):

    def evaluate(self, **kwargs):
        raise NotImplemented("Implement evaluation functions")

    @staticmethod
    def update_policy(policy, dirichlet_belief, utility, l, lr, n_iter, **kwargs):
        raise NotImplemented("Implement evaluation functions")

    def fit(self, **kwargs):
        raise NotImplemented("Implement evaluation functions")


class Algorithm(BaseAlgorithm):

    def __init__(self, true_model, policy, utility):
        self.true_model = true_model
        self.true_model_delta = get_delta(Px_y=self.true_model.Px_y,
                                          Px_yz=self.true_model.Px_yz)
        self.policy = policy
        self.utility = utility
        self.results = None
        self.run_parameters = None

    def evaluate(self, policy, l):
        """
        Evaluate policy on true model
        """
        results = dict()
        results["fairness"] = np.round(get_fairness(policy, self.true_model_delta), 4)
        results["utility"] = get_utility(policy, self.true_model, self.utility)
        results["total"] = (1 - l) * results["utility"] - l * results["fairness"]
        return results

    def save_results(self, save_path):
        np.savetxt(save_path + "/policy.csv", self.policy, delimiter=",")
        self.results.to_csv(save_path + "/results.csv")
        pd.DataFrame(self.run_parameters, index=["parameters"]).to_csv(save_path + "/run_parameters.csv")
