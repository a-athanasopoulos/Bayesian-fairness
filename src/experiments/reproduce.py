"""
The current python script reproduce the results of the "Approximate Inference for the Bayesian Fairness
Framework" paper.

To run the code first setup the directories:
1. base_path: the path of the current repository
   example: base_path = "/Users/my_user/my_projects/Bayesian-fairness"
"""
from src.experiments.continus_bayesian_fairness.non_seq.bootstraping_exp_final import \
    run_continuous_compass_experiment as non_sequential_exp
from src.experiments.continus_bayesian_fairness.seq.bootstraping_exp_seq import \
    run_continuous_compass_experiment as sequential_exp
from src.experiments.discreate_baysian_fairness.non_seq.dis_exp_final import \
    run_discrete_compass_experiment as dis_non_sequential_exp
from src.utils.utils import create_directory

# add your base path
base_path = "/Users/andreasathanasopoulos/Phd/projects/bayesian_fairness/my_code/Bayesian-fairness"
data_path = base_path + "/data"

n_times = 50

# *********************************************************
# A. Continuous experiment

shuffle = True
lr = 0.01
n_iter = 1500
opt_parameters = {
    "n_iter": n_iter,
    "lr": lr,
    "bootstrap_models": 16,
}

# 1. run non sequential continuous experiment
exp_name = "non-sequential"
exp_number = f"adam_lr_{lr}_iter_{n_iter}_s_{shuffle}"
save_path = base_path + f"/results/bayesian_fairness/continuous/{exp_name}/{exp_number}"
create_directory(save_path)
l_list = [0.0, 0.5, 0.8, 1.0]
non_sequential_exp(data_path=data_path,
                   save_path=save_path,
                   dataset_name="compas",
                   opt_parameters=opt_parameters,
                   shuffle=shuffle,
                   l_list=l_list,
                   n_times=n_times)

# 2. run  sequential continuous experiment
exp_name = "sequential"
exp_number = f"adam_lr_{lr}_iter_{n_iter}_s_{shuffle}"
save_path = base_path + f"/my_code/Bayesian-fairness/results/bayesian_fairness/continuous/{exp_name}/{exp_number}"
create_directory(save_path)
l_list = [0.0, 0.5, 0.8, 1.0]
sequential_exp(data_path=data_path,
               save_path=save_path,
               dataset_name="compas",
               opt_parameters=opt_parameters,
               shuffle=True,
               l_list=l_list,
               n_times=n_times)

# *********************************************************
# B. Discrete experiment
shuffle = True
lr = 1.0
n_iter = 1500
opt_parameters = {
    "n_iter": n_iter,
    "lr": lr,
    "bootstrap_models": 16,
}

# 1. non-sequential
exp_name = "non_sequential"
exp_number = f"adam_lr_{lr}_iter_{n_iter}_s_{shuffle}"
save_path = base_path + f"/my_code/Bayesian-fairness/results/bayesian_fairness/discrete/{exp_name}/{exp_number}"
create_directory(save_path)
l_list = [0.0, 0.5, 0.8, 1.0]
dis_non_sequential_exp(data_path=data_path, save_path=save_path, dataset_name="compas")
