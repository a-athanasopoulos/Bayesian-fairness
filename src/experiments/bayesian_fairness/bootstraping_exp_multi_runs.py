from src.experiments.bayesian_fairness.plot_utils import plot_results
from src.experiments.bayesian_fairness.utilts import run_bayesian_algorithm, run_marginal_algorithm, \
    run_bootstrapping_algorithm
from src.discreate.models.dirichlet_model import DirichletModel
from src.discreate.utils.data_utils import get_discrete_compas_dataset
from src.discreate.utils.plot_results import comparison_subplots
from src.discreate.utils.policy import get_random_policy
from src.utils.utils import create_directory
import pandas as pd


def run_compass_experiment(data_path, save_path):
    n_times = 10
    # ******************Load Data*************************
    # set features
    Z_atr = ["sex", "race"]
    X_atr = ['age_cat', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree']
    Y_atr = ['two_year_recid']

    # features to clip
    clip_features = ["juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
    clip_value = 2

    # load dataset
    dataset, (n_x, n_y, n_z) = get_discrete_compas_dataset(data_path=data_path,
                                                           Z_atr=Z_atr,
                                                           X_atr=X_atr,
                                                           Y_atr=Y_atr,
                                                           clip_features=clip_features,
                                                           clip_value=clip_value)
    # split dataset into training test set
    train_data = dataset.iloc[0:6000]
    test_data = dataset.iloc[6000:]

    print("training size:", train_data.shape)
    print("testing size:", test_data.shape)

    # ******************Run Experiment*************************
    # test dirichlet belief
    dirichlet_prior = 0.5
    true_dirichlet_model = DirichletModel(n_x=n_x, n_z=n_z, n_y=n_y, prior=dirichlet_prior)
    true_dirichlet_model.update_posterior_belief(data=test_data)
    true_model = true_dirichlet_model.get_marginal_model()

    num_X, num_Y, num_Z, num_A = n_x, n_y, n_z, 2

    # set parameters
    parameters = {
        "n_iter": 400,
        "lr": 1.0,
        "update_policy_period": 500,
        "n_model": 16,
        "bootstrap_models": 16,
        "l": None
    }
    # l_list = [0.00, 0.10, 0.30, 0.40, 0.50, 0.70, 0.80, 0.90, 1.00, 1.10]
    # l_list = [20.00, 30.00, 40.00, 50.00]
    l_list = [200.00, 250.0, 400.00, 500.0, 1000.0]  # lambda
    # iterate over different l parameters
    for l in l_list:
        parameters["l"] = l
        print(f"run experiment for l : {l}")
        tmp_l_save_path = save_path + f"/l_{l}"

        bootstrapping_results = []
        marginal_results = []
        bayesian_results = []
        for run in range(n_times):
            initial_policy = get_random_policy(size=(num_A, num_X))
            tmp_save_path = tmp_l_save_path + f"/run_{run}"
            print(f"run bootstrapping_fair_algorithm")
            bootstrapping_fair_algorithm = run_bootstrapping_algorithm(train_data=train_data,
                                                                       initial_policy=initial_policy,
                                                                       true_model=true_model,
                                                                       num_X=num_X,
                                                                       num_Y=num_Y,
                                                                       num_Z=num_Z,
                                                                       num_A=num_A,
                                                                       prior=dirichlet_prior,
                                                                       alg_parameters=parameters,
                                                                       save_path=tmp_save_path)
            bootstrapping_results += [bootstrapping_fair_algorithm.results]

            print(f"run marginal_fair_algorithm")
            marginal_fair_algorithm = run_marginal_algorithm(train_data=train_data,
                                                             initial_policy=initial_policy,
                                                             true_model=true_model,
                                                             num_X=num_X,
                                                             num_Y=num_Y,
                                                             num_Z=num_Z,
                                                             num_A=num_A,
                                                             prior=dirichlet_prior,
                                                             alg_parameters=parameters,
                                                             save_path=tmp_save_path)
            marginal_results += [marginal_fair_algorithm.results]
            print(f"run bayesian_fair_algorithm")
            bayesian_fair_algorithm = run_bayesian_algorithm(train_data=train_data,
                                                             initial_policy=initial_policy,
                                                             true_model=true_model,
                                                             num_X=num_X,
                                                             num_Y=num_Y,
                                                             num_Z=num_Z,
                                                             num_A=num_A,
                                                             prior=dirichlet_prior,
                                                             alg_parameters=parameters,
                                                             save_path=tmp_save_path)
            bayesian_results += [bayesian_fair_algorithm.results]

            results_list = [marginal_fair_algorithm.results,
                            bayesian_fair_algorithm.results,
                            bootstrapping_fair_algorithm.results]

            labels = ["marginal",
                      "bayes",
                      "bootstrap"]

            plot_results(results=results_list,
                         labels=labels,
                         save_path=tmp_save_path,
                         show=0)

            comparison_subplots(results=results_list,
                                keys=["total", "utility", "fairness"],
                                labels=labels,
                                save_path=tmp_save_path,
                                show=0)
        save_name = tmp_l_save_path + "/boostrap_results_all.csv"
        pd.concat(bootstrapping_results, keys=[f"run_{i}" for i in range(n_times)], axis=1).to_csv(save_name)

        save_name = tmp_l_save_path + "/marginal_results_all.csv"
        pd.concat(marginal_results, keys=[f"run_{i}" for i in range(n_times)], axis=1).to_csv(save_name)

        save_name = tmp_l_save_path + "/bayesian_results_all.csv"
        pd.concat(bayesian_results, keys=[f"run_{i}" for i in range(n_times)], axis=1).to_csv(save_name)


if __name__ == "__main__":
    # ******************PATH Configuration****************
    exp_name = "exp_compas_boostrap"
    exp_number = "final"  # add experiment sub-name
    base_path = "/Users/andreasathanasopoulos/Phd/projects/bayesian_fairness/"  # add your base path
    data_path = base_path + "/my_code/Bayesian-fairness/data"
    save_path = base_path + f"/my_code/Bayesian-fairness/results/bayesian_fairness/discrete/{exp_name}/{exp_number}"

    # create exp directory
    create_directory(save_path)
    # ******************Run experiment****************
    run_compass_experiment(data_path=data_path, save_path=save_path)
