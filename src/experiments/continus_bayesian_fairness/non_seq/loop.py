import pandas as pd

from src.continuous.algorithm.marginal_fair_optimization import ContinuousMarginalFairOptimization, \
    ContinuousBootstrapFairOptimization
from src.utils.utils import create_directory


def run_marginal_algorithm(train_data,
                           test_data_and_models,
                           Z_atr,
                           X_atr,
                           Y_atr,
                           n_y,
                           n_z,
                           initial_policy_weights,
                           prior,
                           opt_parameters,
                           update_policy_period,
                           save_path):
    horizon = train_data.shape[0] // update_policy_period

    save_path = save_path + "/marginal"
    # inner loop
    learning_rate = opt_parameters["lr"]
    results = []
    policy_weights = initial_policy_weights
    for step in range(horizon):
        data_till = (step + 1) * update_policy_period
        marginal_fair_algorithm = ContinuousMarginalFairOptimization(train_data=train_data.iloc[0:data_till],
                                                                     test_data_and_models=test_data_and_models,
                                                                     Z_atr=Z_atr,
                                                                     X_atr=X_atr,
                                                                     Y_atr=Y_atr,
                                                                     n_y=n_y,
                                                                     n_z=n_z,
                                                                     prior=prior)

        marginal_fair_algorithm.compile(initial_policy_weights=policy_weights,
                                        learning_rate=learning_rate)
        marginal_fair_algorithm.fit(**opt_parameters)
        create_directory(save_path + f"/opt_results/step_{step}")
        marginal_fair_algorithm.save_results(save_path=save_path + f"/opt_results/step_{step}")
        marginal_fair_algorithm.plot_history(save_path=save_path + f"/opt_results/step_{step}")
        policy_weights = marginal_fair_algorithm.final_policy_weights
        results += [marginal_fair_algorithm.results.iloc[-1]]

    results = pd.concat(results, axis=1, ignore_index=True).T
    results.to_csv(save_path + f"/results.csv")
    return results


def run_bootstrap_algorithm(train_data,
                            test_data_and_models,
                            Z_atr,
                            X_atr,
                            Y_atr,
                            n_y,
                            n_z,
                            initial_policy_weights,
                            prior,
                            opt_parameters,
                            update_policy_period,
                            save_path):
    horizon = train_data.shape[0] // update_policy_period
    save_path = save_path + "/bootstrap"
    # inner loop
    learning_rate = opt_parameters["lr"]
    results = []
    policy_weights = initial_policy_weights
    for step in range(horizon):

        data_till = (step + 1) * update_policy_period
        print(f"step {step} with data from {0} to {data_till}")
        bootstrap_fair_algorithm = ContinuousBootstrapFairOptimization(train_data=train_data.iloc[0:data_till],
                                                                       test_data_and_models=test_data_and_models,
                                                                       Z_atr=Z_atr,
                                                                       X_atr=X_atr,
                                                                       Y_atr=Y_atr,
                                                                       n_y=n_y,
                                                                       n_z=n_z,
                                                                       prior=prior)

        bootstrap_fair_algorithm.compile(initial_policy_weights=policy_weights,
                                         learning_rate=learning_rate)
        bootstrap_fair_algorithm.fit(**opt_parameters)
        create_directory(save_path + f"/opt_results/step_{step}")
        bootstrap_fair_algorithm.save_results(save_path=save_path + f"/opt_results/step_{step}")
        bootstrap_fair_algorithm.plot_history(save_path=save_path + f"/opt_results/step_{step}")
        policy_weights = bootstrap_fair_algorithm.final_policy_weights
        results += [bootstrap_fair_algorithm.results.iloc[-1]]

    results = pd.concat(results, axis=1, ignore_index=True).T
    results.to_csv(save_path + f"/results.csv")
    return results
