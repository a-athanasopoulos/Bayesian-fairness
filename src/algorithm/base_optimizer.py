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

    def fit(self, train_data, belief, update_policy_period, l, lr, n_iter, n_model=None, **kwargs):
        horizon = train_data.shape[0]
        steps = horizon // update_policy_period

        self.run_parameters = {
            "update_policy_period": update_policy_period,
            "l": l,
            "lr": lr,
            "n_iter": n_iter,
            "n_model": n_model
        }

        self.results = []
        for step in range(steps):
            # update policy step
            self.policy = self.update_policy(policy=self.policy,
                                             dirichlet_belief=belief,
                                             utility=self.utility,
                                             l=l,
                                             lr=lr,
                                             n_iter=n_iter,
                                             n_model=n_model)  # SDG to update policy

            # evaluation step
            step_results = self.evaluate(policy=self.policy,
                                         l=l)
            self.results += [step_results]

            # update belief step
            data_start_index = step * update_policy_period
            data_stop_index = min(data_start_index + update_policy_period, horizon)
            belief.update_posterior_belief(train_data.iloc[data_start_index: data_stop_index])

            # print progress
            print(f"--- Step : {data_start_index + 1} \n  ------- {step_results}")
        self.results = pd.DataFrame(self.results)

    def save_results(self, save_path):
        np.savetxt(save_path + "/policy.csv", self.policy, delimiter=",")
        self.results.to_csv(save_path + "/results.csv")
        pd.DataFrame(self.run_parameters, index=["parameters"]).to_csv(save_path + "/run_parameters.csv")
