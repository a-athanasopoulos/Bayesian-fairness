from src.algorithm.bayes_fair_optimization import BayesianFairOptimization
from src.algorithm.bootstrap_fair_optimization import BootstrapFairOptimization
from src.algorithm.marginal_fair_optimization import MarginalFairOptimization
from src.models.dirichlet_model import DirichletModel
from src.utils.plot_results import comparison_plots
from src.utils.utility import get_eye_utility
from src.utils.utils import create_directory


def run_marginal_algorithm(train_data,
                           initial_policy,
                           true_model,
                           num_X,
                           num_Y,
                           num_Z,
                           num_A,
                           prior,
                           alg_parameters,
                           save_path):
    train_belief = DirichletModel(n_x=num_X, n_y=num_Y, n_z=num_Z, prior=prior)
    utility = get_eye_utility(size=num_A)

    marginal_fair_algorithm = MarginalFairOptimization(true_model=true_model,
                                                       policy=initial_policy,
                                                       utility=utility)

    marginal_fair_algorithm.fit(train_data=train_data,
                                belief=train_belief,
                                **alg_parameters)

    create_directory(save_path + "/marginal")

    marginal_fair_algorithm.save_results(save_path=save_path + "/marginal")
    return marginal_fair_algorithm


def run_bayesian_algorithm(train_data,
                           initial_policy,
                           true_model,
                           num_X,
                           num_Y,
                           num_Z,
                           num_A,
                           prior,
                           alg_parameters,
                           save_path):
    utility = get_eye_utility(size=num_A)

    train_belief = DirichletModel(n_x=num_X, n_y=num_Y, n_z=num_Z, prior=prior)

    bayesian_fair_algorithm = BayesianFairOptimization(true_model=true_model, policy=initial_policy, utility=utility)

    bayesian_fair_algorithm.fit(train_data=train_data,
                                belief=train_belief,
                                **alg_parameters)

    create_directory(save_path + "/bayes")

    bayesian_fair_algorithm.save_results(save_path=save_path + "/bayes")
    return bayesian_fair_algorithm


def run_bootstrapping_algorithm(train_data,
                                initial_policy,
                                true_model,
                                num_X,
                                num_Y,
                                num_Z,
                                num_A,
                                prior,
                                alg_parameters,
                                save_path):
    dim_xyza = (num_X, num_Y, num_Z, num_A)
    utility = get_eye_utility(size=num_A)

    bayesian_fair_algorithm = BootstrapFairOptimization(true_model=true_model, policy=initial_policy, utility=utility)

    bayesian_fair_algorithm.fit(train_data=train_data,
                                dim_xyza=dim_xyza,
                                **alg_parameters)

    create_directory(save_path + "/bootstrapping")

    bayesian_fair_algorithm.save_results(save_path=save_path + "/bootstrapping")
    return bayesian_fair_algorithm
