from src.discrete.utils.plot_results import comparison_plots
import matplotlib.pyplot as plt


def plot_results(results, labels, save_path, show=0):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, )

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
