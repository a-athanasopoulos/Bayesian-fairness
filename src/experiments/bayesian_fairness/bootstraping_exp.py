from src.experiments.bayesian_fairness.plot_utils import plot_results
from src.experiments.bayesian_fairness.utilts import run_bayesian_algorithm, run_marginal_algorithm, \
    run_bootstrapping_algorithm
from src.models.dirichlet_model import DirichletModel
from src.utils.data_utils import get_discrete_compas_dataset
from src.utils.plot_results import comparison_subplots
from src.utils.policy import get_random_policy
from src.utils.utils import create_directory


def run_compass_experiment(data_path, save_path):
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
        "update_policy_period": 100,
        "n_model": 16,
        "bootstrap_models": 64,
        "l": None
    }
    l_list = [0.0, 0.5, 1.0]  # lambda
    initial_policy = get_random_policy(size=(num_A, num_X))
    # iterate over different l parameters
    for l in l_list:
        print(f"run experiment for l : {l}")
        parameters["l"] = l
        tmp_save_path = save_path + f"/l_{l}"
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


if __name__ == "__main__":
    # ******************PATH Configuration****************
    exp_name = "exp_compas_boostrap"
    exp_number = "exp_1_64"  # add experiment sub-name
    base_path = "/Users/andreasathanasopoulos/Phd/projects/bayesian_fairness/"  # add your base path
    data_path = base_path + "/my_code/Bayesian-fairness/data"
    save_path = base_path + f"/my_code/Bayesian-fairness/results/bayesian_fairness/{exp_name}/{exp_number}"

    # create exp directory
    create_directory(save_path)
    # ******************Run experiment****************
    run_compass_experiment(data_path=data_path, save_path=save_path)
