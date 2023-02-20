import pandas as pd
import os

def merge_results(results_path):
    subfolders = [f.path for f in os.scandir(results_path) if f.is_dir()]

    bootstrap_results = []
    marginal_results = []
    for run_path in subfolders:
        bootstrap_results += [pd.read_csv(run_path+ "/bootstrap_results.csv")]
        marginal_results += [pd.read_csv(run_path+ "/marginal_results.csv")]

    save_name = results_path + "/boostrap_results_all.csv"
    pd.concat(bootstrap_results,
              keys=[f"run_{i}" for i in range(len(bootstrap_results))], axis=1).to_csv(save_name)

    save_name = results_path + "/marginal_results_all.csv"
    pd.concat(marginal_results,
              keys=[f"run_{i}" for i in range(len(marginal_results))], axis=1).to_csv(save_name)


if __name__ == "__main__":
    l = 1.0
    exp_name = "seq_exp_compass_hyperparameter_tuning"
    exp_number = "adam_lr_0.01_iter_1500_s_True"  # add experiment sub-name
    base_path = "/Users/andreasathanasopoulos/Phd/projects/bayesian_fairness/"  # add your base path
    results_path = base_path + f"/my_code/Bayesian-fairness/results/bayesian_fairness/continuous/{exp_name}/{exp_number}/l_{l}"
    merge_results(results_path)