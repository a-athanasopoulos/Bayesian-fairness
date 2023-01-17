import matplotlib.pyplot as plt


def comparison_plots(marginal_results, bayesian_results, atribure, title, save_path=None, show=1):
    plt.figure()
    plt.plot(marginal_results[atribure], label="marginal")
    plt.plot(bayesian_results[atribure], label="bayes")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path + f"/compare_{atribure}.png")
    if show:
        plt.show()

    plt.close()
