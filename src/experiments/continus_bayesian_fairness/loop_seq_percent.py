import pandas as pd
import numpy as np
from src.continuous.algorithm.marginal_fair_optimization import ContinuousMarginalFairOptimization, \
    ContinuousBootstrapFairOptimization
from src.continuous.tf.models.logistic_regression import LogisticRegressionTF
from src.utils.utils import create_directory


def run_marginal_algorithm_seq(train_data,
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
    percent = 1.0
    accepted_per_step = int(np.ceil(update_policy_period * percent))
    horizon = train_data.shape[0] // update_policy_period

    save_path = save_path + "/marginal"
    # inner loop
    learning_rate = opt_parameters["lr"]
    results = []
    policy_weights = initial_policy_weights
    accepted_mask = pd.Series([False] * train_data.shape[0])

    for step in range(horizon):
        data_from = (step) * update_policy_period
        data_till = (step + 1) * update_policy_period

        policy = LogisticRegressionTF(input_dim=len(X_atr))
        policy.set_weights(policy_weights)
        predictions = policy.predict(train_data.iloc[data_from:data_till][X_atr])
        accepted_ind = np.argsort(predictions.ravel())[::-1][0:accepted_per_step] + data_from
        accepted_mask[accepted_ind] = True
        print(sum(predictions))
        marginal_fair_algorithm = ContinuousMarginalFairOptimization(train_data=train_data[accepted_mask],
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


def run_bootstrap_algorithm_seq(train_data,
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
    percent = 1.0
    accepted_per_step = int(np.ceil(update_policy_period * percent))
    horizon = train_data.shape[0] // update_policy_period
    save_path = save_path + "/bootstrap"
    # inner loop
    learning_rate = opt_parameters["lr"]
    results = []
    policy_weights = initial_policy_weights

    accepted_mask = pd.Series([False] * train_data.shape[0])
    accepted_numbers = {}
    for step in range(horizon):
        data_from = (step) * update_policy_period
        data_till = (step + 1) * update_policy_period

        policy = LogisticRegressionTF(input_dim=len(X_atr))
        policy.set_weights(policy_weights)
        predictions = policy.predict(train_data.iloc[data_from:data_till][X_atr])
        accepted_ind = np.argsort(predictions.ravel())[::-1][0:accepted_per_step] + data_from
        accepted_mask[accepted_ind] = True
        accepted_numbers[step] = sum(predictions)
        print(f"step {step + 1} accepted: ", accepted_ind.shape[0])
        print(f"total accepted members: {train_data[accepted_mask].shape[0]}")
        bootstrap_fair_algorithm = ContinuousBootstrapFairOptimization(train_data=train_data[accepted_mask],
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
    print(f"step  accepted % :", accepted_mask.mean())
    results = pd.concat(results, axis=1, ignore_index=True).T
    results.to_csv(save_path + f"/results.csv")
    pd.DataFrame(accepted_numbers).to_csv(save_path + f"/accepted.csv")
    return results
