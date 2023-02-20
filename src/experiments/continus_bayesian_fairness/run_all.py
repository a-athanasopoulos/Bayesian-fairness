from src.experiments.continus_bayesian_fairness.bootstraping_exp_final import \
    run_continuous_compass_experiment as no_seq_exp
from src.experiments.continus_bayesian_fairness.bootstraping_exp_seq import run_continuous_compass_experiment as seq_exp
from src.utils.utils import create_directory

base_path = "/Users/andreasathanasopoulos/Phd/projects/bayesian_fairness/"  # add your base path
data_path = base_path + "/my_code/Bayesian-fairness/data"

n_times = 50
l_list = [1.0, 0.0]
for shuffle in [True]:
    exp_name = "exp_compass_hyperparameter_tuning"
    for lr in [0.01]:
        for iter in [1500]:
            opt_parameters = {
                "n_iter": iter,
                "lr": lr,
                "bootstrap_models": 16,
                "lambda_parameter": None
            }
            exp_number = f"adam_lr_{lr}_iter_{iter}_s_{shuffle}"
            save_path = base_path + f"/my_code/Bayesian-fairness/results/bayesian_fairness/continuous/{exp_name}/{exp_number}"
            create_directory(save_path)
            no_seq_exp(data_path=data_path,
                       save_path=save_path,
                       dataset_name="compas",
                       opt_parameters=opt_parameters,
                       shuffle=shuffle,
                       l_list=l_list,
                       n_times=n_times)
    #
    # exp_name = "seq_exp_compass_hyperparameter_tuning"
    # for lr in [0.01]:
    #     for iter in [1500]:
    #         opt_parameters = {
    #             "n_iter": iter,
    #             "lr": lr,
    #             "bootstrap_models": 16,
    #             "lambda_parameter": None
    #         }
    #         exp_number = f"adam_lr_{lr}_iter_{iter}_s_{shuffle}"
    #         save_path = base_path + f"/my_code/Bayesian-fairness/results/bayesian_fairness/continuous/{exp_name}/{exp_number}"
    #         create_directory(save_path)
    #         seq_exp(data_path=data_path,
    #                 save_path=save_path,
    #                 dataset_name="compas",
    #                 opt_parameters=opt_parameters,
    #                 shuffle=True,
    #                 l_list=l_list,
    #                 n_times=n_times)
