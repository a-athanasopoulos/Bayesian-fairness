from src.discreate.tf.models.logistic_regression import LogisticRegressionTF
from src.discreate.models.dirichlet_model import DirichletModel
from src.discreate.utils.data_utils import get_discrete_compas_dataset
from src.discreate.utils.plot_results import comparison_subplots
from src.experiments.discreate_baysian_fairness.non_seq.loop import run_bootstrap_algorithm
from src.experiments.discreate_baysian_fairness.non_seq.loop import run_marginal_algorithm
from src.experiments.discreate_baysian_fairness.non_seq.loop import run_bayesian_algorithm

from src.experiments.discreate_baysian_fairness.plot_utils import plot_results
from src.utils.utils import create_directory


def run_discrete_compass_experiment(data_path,
                                    save_path,
                                    dataset_name,
                                    n_times,
                                    l_list,
                                    opt_parameters,
                                    shuffle=True):
    # ******************Load Data*************************

    # load dataset
    if dataset_name == "compas":
        train_data, test_data, X_atr, Y_atr, Z_atr, n_x, n_y, n_z = get_discrete_compas_dataset(data_path=data_path)

    print("training size:", train_data.shape)
    print("testing size:", test_data.shape)

    # ******************Run Experiment*************************
    dirichlet_prior = 0.5
    test_dirichlet_model = DirichletModel(n_x=n_x, n_z=n_z, n_y=n_y, prior=dirichlet_prior)
    test_dirichlet_model.update_posterior_belief(data=test_data)
    test_model = test_dirichlet_model.get_marginal_model()

    # set parameters
    update_policy_period = 500

    initial_policy_weights_list = [LogisticRegressionTF(input_dim=n_x).get_weights() for _ in range(n_times)]
    # iterate over different l parameters

    for tmp_l in l_list:
        print(f"run experiment for l : {tmp_l}")
        opt_parameters["lambda_parameter"] = tmp_l
        tmp_l_save_path = save_path + f"/l_{tmp_l}"

        bootstrapping_results = []
        marginal_results = []
        bayes_results = []
        for i in range(n_times):
            if shuffle:
                train_data = train_data.sample(frac=1, replace=False).reset_index(drop=True)
            tmp_save_path = tmp_l_save_path + f"/run_{i}"

            print(f"run bootstrapping_fair_algorithm")
            bootstrap_cont_results = run_bootstrap_algorithm(train_data=train_data,
                                                             test_model=test_model,
                                                             Z_atr=Z_atr,
                                                             X_atr=X_atr,
                                                             Y_atr=Y_atr,
                                                             n_x=n_x,
                                                             n_y=n_y,
                                                             n_z=n_z,
                                                             initial_policy_weights=initial_policy_weights_list[i],
                                                             prior=dirichlet_prior,
                                                             update_policy_period=update_policy_period,
                                                             opt_parameters=opt_parameters,
                                                             save_path=tmp_save_path)
            bootstrap_cont_results.to_csv(tmp_save_path + "/bootstrap_results.csv")
            bootstrapping_results += [bootstrap_cont_results]

            print(f"run bayesian_fair_algorithm")
            bayes_cont_results = run_bayesian_algorithm(train_data=train_data,
                                                        test_model=test_model,
                                                        Z_atr=Z_atr,
                                                        X_atr=X_atr,
                                                        Y_atr=Y_atr,
                                                        n_x=n_x,
                                                        n_y=n_y,
                                                        n_z=n_z,
                                                        initial_policy_weights=initial_policy_weights_list[i],
                                                        prior=dirichlet_prior,
                                                        update_policy_period=update_policy_period,
                                                        opt_parameters=opt_parameters,
                                                        save_path=tmp_save_path)
            bayes_cont_results.to_csv(tmp_save_path + "/bayes_results.csv")
            bayes_results += [bayes_cont_results]

            print(f"run marginal_fair_algorithm")
            marginal_cont_results = run_marginal_algorithm(train_data=train_data,
                                                           test_model=test_model,
                                                           Z_atr=Z_atr,
                                                           X_atr=X_atr,
                                                           Y_atr=Y_atr,
                                                           n_x=n_x,
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
                            marginal_cont_results,
                            bayes_cont_results]

            labels = ["bootstrap",
                      "marginal",
                      "bayes"]

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

        save_name = tmp_l_save_path + "/bayes_results_all.csv"
        pd.concat(bayes_results, keys=[f"run_{i}" for i in range(n_times)], axis=1).to_csv(save_name)

        save_name = tmp_l_save_path + "/marginal_results_all.csv"
        pd.concat(marginal_results, keys=[f"run_{i}" for i in range(n_times)], axis=1).to_csv(save_name)


if __name__ == "__main__":
    # ******************PATH Configuration****************
    exp_name = "exp_compass_tests"
    exp_number = "test_2_shuffle_lr_0.01_1500"  # add experiment sub-name
    base_path = "/Users/andreasathanasopoulos/Phd/projects/bayesian_fairness/"  # add your base path
    data_path = base_path + "/my_code/Bayesian-fairness/data"
    save_path = base_path + f"/my_code/Bayesian-fairness/results/bayesian_fairness/discrete_new/{exp_name}/{exp_number}"

    # create exp directory
    create_directory(save_path)
    # ******************Run experiment****************
    n_times = 50
    shuffle = True
    lr = 1.0
    n_iter = 1500
    opt_parameters = {
        "n_iter": n_iter,
        "lr": lr,
        "bootstrap_models": 16,
    }
    l_list = [0.0, 0.5, 1.0]
    run_discrete_compass_experiment(data_path=data_path,
                                    save_path=save_path,
                                    dataset_name="compas",
                                    n_times=n_times,
                                    shuffle=shuffle,
                                    opt_parameters=opt_parameters,
                                    l_list=l_list)
