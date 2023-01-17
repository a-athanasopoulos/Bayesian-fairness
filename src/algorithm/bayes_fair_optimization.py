from src.algorithm.base_optimizer import Algorithm
from src.loss.fairness import get_fairness_gradient
from src.loss.utility import get_utility_gradient
from src.utils.gradient import project_gradient
from src.utils.model import get_delta
from src.utils.policy import normalize_policy


class BayesianFairOptimization(Algorithm):

    @staticmethod
    def update_policy(policy, dirichlet_belief, utility, l, lr, n_iter, n_model, **kwargs):
        """
        Marginal Policy Dirichlet
        """
        models = []
        model_delta = []
        for m in range(n_model):
            model = dirichlet_belief.sample_model()
            models += [model]
            model_delta += [get_delta(model.Px_y, model.Px_yz)]

        for i in range(n_iter):
            tmp_index = i % n_model
            tmp_model = models[tmp_index]
            tmp_delta = model_delta[tmp_index]
            fairness_gradient = get_fairness_gradient(policy, tmp_delta)
            utility_gradient = get_utility_gradient(tmp_model, utility)
            gradient = (1 - l) * utility_gradient + l * fairness_gradient  # minus on the gradient calc.
            gradient = project_gradient(gradient)
            policy = policy + lr * gradient  # maximize Utility & minimize fairness constrain.
            policy = normalize_policy(policy)

        return policy
