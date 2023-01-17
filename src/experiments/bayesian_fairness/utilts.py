from src.utils.plot_results import comparison_plots


def plot_results(marginal_fair_algorithm, bayesian_fair_algorithm, save_path, show=0):
    comparison_plots(marginal_results=marginal_fair_algorithm.results,
                     bayesian_results=bayesian_fair_algorithm.results,
                     atribure="utility",
                     title="Utility U",
                     save_path=save_path,
                     show=show)

    # %%

    comparison_plots(marginal_results=marginal_fair_algorithm.results,
                     bayesian_results=bayesian_fair_algorithm.results,
                     atribure="fairness",
                     title="Fairness F",
                     save_path=save_path,
                     show=show)

    # %%

    comparison_plots(marginal_results=marginal_fair_algorithm.results,
                     bayesian_results=bayesian_fair_algorithm.results,
                     atribure="total",
                     title="Total T",
                     save_path=save_path,
                     show=show)
