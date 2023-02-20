from src.discreate.utils.plot_results import comparison_plots
import matplotlib.pyplot as plt


def plot_results(results, labels, save_path, show=0):

    comparison_plots(results=results,
                     labels=labels,
                     atribure="eval_utility",
                     title="Utility U",
                     save_path=save_path,
                     show=show)

    # %%

    comparison_plots(results=results,
                     labels=labels,
                     atribure="eval_fairness_loss",
                     title="Fairness F",
                     save_path=save_path,
                     show=show)

    # %%

    comparison_plots(results=results,
                     labels=labels,
                     atribure="eval_loss",
                     title="Total T",
                     save_path=save_path,
                     show=show)
