import numpy as np

from src.continuous.models.models import get_models_from_data
from src.continuous.tf.models.logistic_regression import LogisticRegressionTF
from src.continuous.utils.adult_dataset import get_adult_datasets
from src.continuous.utils.data_utils import get_continuous_compas_dataset
from src.continuous.utils.plot_results import comparison_subplots
from src.experiments.continus_bayesian_fairness.plot_utils import plot_results
from src.experiments.continus_bayesian_fairness.loop_1 import run_marginal_algorithm, run_bootstrap_algorithm
from src.utils.utils import create_directory


def run_continuous_compass_experiment(data_path, save_path, dataset_name):
    # ******************Load Data*************************
    n_times = 10

    # load dataset
    if dataset_name == "compas":
        train_data, test_data, X_atr, Y_atr, Z_atr, n_y, n_z = get_continuous_compas_dataset(data_path=data_path)
    elif dataset_name == "adult":
        train_data, test_data, X_atr, Y_atr, Z_atr, n_y, n_z = get_adult_datasets(data_path=data_path)

    print("training size:", train_data.shape)
    print("testing size:", test_data.shape)

    # ******************Run Experiment*************************

    # set parameters
    opt_parameters = {
        "n_iter": 500,
        "lr": 0.5,
        "bootstrap_models": 16,
        "lambda_parameter": None
    }

    update_policy_period = 500
    l_list = [0.30, 0.40,  0.50, 0.80, 0.90, 1.00]  # lambda
    dirichlet_prior = 0.5
    initial_policy_weights_list = [LogisticRegressionTF(input_dim=len(X_atr)).get_weights() for i in range(n_times)]
    # iterate over different l parameters

    test_model = get_models_from_data(data=test_data,
                                      X_atr=X_atr, Y_atr=Y_atr, Z_atr=Z_atr,
                                      n_y=n_y, n_z=n_z)

    test_data_and_models = (test_data[X_atr].values,
                            np.reshape(test_data[Y_atr].values, (-1, 1)),
                            test_model[0],
                            test_model[1],
                            test_model[2],
                            test_model[3])

    for tmp_l in l_list:
        print(f"run experiment for l : {tmp_l}")
        opt_parameters["lambda_parameter"] = tmp_l
        tmp_l_save_path = save_path + f"/l_{tmp_l}"

        bootstrapping_results = []
        marginal_results = []
        for i in range(n_times):
            tmp_save_path = tmp_l_save_path + f"/run_{i}"

            print(f"run bootstrapping_fair_algorithm")
            bootstrap_cont_results = run_bootstrap_algorithm(train_data=train_data,
                                                             test_data_and_models=test_data_and_models,
                                                             Z_atr=Z_atr,
                                                             X_atr=X_atr,
                                                             Y_atr=Y_atr,
                                                             n_y=n_y,
                                                             n_z=n_z,
                                                             initial_policy_weights=initial_policy_weights_list[i],
                                                             prior=dirichlet_prior,
                                                             update_policy_period=update_policy_period,
                                                             opt_parameters=opt_parameters,
                                                             save_path=tmp_save_path)
            bootstrap_cont_results.to_csv(tmp_save_path + "/bootstrap_results.csv")
            bootstrapping_results += [bootstrap_cont_results]

            print(f"run marginal_fair_algorithm")
            marginal_cont_results = run_marginal_algorithm(train_data=train_data,
                                                           test_data_and_models=test_data_and_models,
                                                           Z_atr=Z_atr,
                                                           X_atr=X_atr,
                                                           Y_atr=Y_atr,
                                                           n_y=n_y,
                                                           n_z=n_z,
                                                           initial_policy_weights=initial_policy_weights_list[i],
                                                           prior=dirichlet_prior,
                                                           update_policy_period=update_policy_period,
                                                           opt_parameters=opt_parameters,
                                                           save_path=tmp_save_path)
            marginal_cont_results.to_csv(tmp_save_path + "/marginal_results.csv")
            marginal_results += [marginal_cont_results]

            import pandas as pd
            last_results = pd.concat([bootstrap_cont_results.iloc[-1],
                                      marginal_cont_results.iloc[-1]], axis=0)
            last_results.to_csv(tmp_save_path + "/last_step_results.csv")

            results_list = [bootstrap_cont_results,
                            marginal_cont_results]

            labels = ["bootstrap",
                      "marginal"]

            plot_results(results=results_list,
                         labels=labels,
                         save_path=tmp_save_path,
                         show=0)

            comparison_subplots(results=results_list,
                                keys=["eval_loss", "eval_utility", "eval_fairness_loss"],
                                labels=labels,
                                save_path=tmp_save_path,
                                show=0)

        save_name = tmp_l_save_path + "/boostrap_results_all.csv"
        pd.concat(bootstrapping_results, keys=[f"run_{i}" for i in range(n_times)], axis=1).to_csv(save_name)

        save_name = tmp_l_save_path + "/marginal_results_all.csv"
        pd.concat(marginal_results, keys=[f"run_{i}" for i in range(n_times)], axis=1).to_csv(save_name)


if __name__ == "__main__":
    # ******************PATH Configuration****************
    exp_name = "exp_compas_boostrap_continuous_final"
    exp_number = "loss_org_final"  # add experiment sub-name
    base_path = "/Users/andreasathanasopoulos/Phd/projects/bayesian_fairness/"  # add your base path
    data_path = base_path + "/my_code/Bayesian-fairness/data"
    save_path = base_path + f"/my_code/Bayesian-fairness/results/bayesian_fairness/continuous/{exp_name}/{exp_number}"

    # create exp directory
    create_directory(save_path)
    # ******************Run experiment****************
    run_continuous_compass_experiment(data_path=data_path, save_path=save_path, dataset_name="compas")
