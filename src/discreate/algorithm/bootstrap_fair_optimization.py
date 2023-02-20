import pandas as pd
from src.discreate.algorithm.base_optimizer import Algorithm
from src.discreate.loss.fairness import get_fairness_gradient
from src.discreate.loss.utility import get_utility_gradient
from src.discreate.models.boostrap_model import BoostrapModel
from src.discreate.utils.gradient import project_gradient
from src.discreate.utils.model import get_delta
from src.discreate.utils.policy import normalize_policy


class BootstrapFairOptimization(Algorithm):
    """
    Bootstrap Fair Optimization
    """

    @staticmethod
    def update_policy(policy, dim, training_data, utility, l, lr, n_iter, n_model, **kwargs):
        """
        Bootstrap update policy
        """

        # sample models
        models = []
        model_delta = []
        for m in range(n_model):
            model = BoostrapModel(n_x=dim[0], n_y=dim[1], n_z=dim[2]).sample_model(training_data)
            models += [model]
            model_delta += [get_delta(model.Px_y, model.Px_yz, model.Pz_y)]

        for i in range(n_iter):
            tmp_index = i % n_model
            tmp_model = models[tmp_index]
            tmp_delta = model_delta[tmp_index]
            fairness_gradient = get_fairness_gradient(policy, tmp_delta)
            utility_gradient = get_utility_gradient(tmp_model, utility)
            gradient = utility_gradient + l * fairness_gradient  # minus on the gradient calc.
            gradient = project_gradient(gradient)
            policy = policy + lr * gradient  # maximize Utility & minimize fairness constrain.
            policy = normalize_policy(policy)

        return policy

    def fit(self, train_data, update_policy_period, dim_xyza, l, lr, n_iter, bootstrap_models=None, **kwargs):
        horizon = train_data.shape[0]
        steps = horizon // update_policy_period

        self.run_parameters = {
            "update_policy_period": update_policy_period,
            "l": l,
            "lr": lr,
            "n_iter": n_iter,
            "n_model": bootstrap_models
        }

        self.results = []
        for step in range(steps):
            data_until = (step + 1) * update_policy_period
            # update policy step
            self.policy = self.update_policy(policy=self.policy,
                                             dim=dim_xyza,
                                             training_data=train_data.iloc[:(step) * update_policy_period],
                                             utility=self.utility,
                                             l=l,
                                             lr=lr,
                                             n_iter=n_iter,
                                             n_model=bootstrap_models)  # SDG to update policy

            # evaluation step
            step_results = self.evaluate(policy=self.policy,
                                         l=l)
            self.results += [step_results]

            # update belief step
            data_start_index = step * update_policy_period

            # print progress
            print(f"--- Step : {data_start_index + 1} \n  ------- {step_results}")
        self.results = pd.DataFrame(self.results)
