from src.algorithm.base_optimizer import Algorithm
from src.loss.fairness import get_fairness_gradient
from src.loss.utility import get_utility_gradient
from src.utils.gradient import project_gradient
from src.utils.model import get_delta
from src.utils.policy import normalize_policy


class MarginalFairOptimization(Algorithm):

    @staticmethod
    def update_policy(policy, dirichlet_belief, utility, l, lr, n_iter, **kwargs):
        """
        Marginal Policy Dirichlet
        """
        model = dirichlet_belief.get_marginal_model()
        model_delta = get_delta(model.Px_y, model.Px_yz)

        for i in range(n_iter):
            fairness_gradient = get_fairness_gradient(policy, model_delta)
            utility_gradient = get_utility_gradient(model, utility)
            gradient = (1 - l) * utility_gradient + l * fairness_gradient  # minus on the gradient calc.
            gradient = project_gradient(gradient)
            policy = policy + lr * gradient  # maximize Utility & minimize fairness constrain.
            policy = normalize_policy(policy)

        return policy
