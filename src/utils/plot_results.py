import matplotlib.pyplot as plt


def comparison_plots(results, labels, atribure, title, save_path=None, show=1):
    plt.figure()
    for result, label in zip(results, labels):
        plt.plot(result[atribure], label=label)
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path + f"/compare_{atribure}.png")
    if show:
        plt.show()

    plt.close()


def comparison_subplots(results, keys, labels, save_path=None, show=1):
    fig, (ax1, ax2, ax3) = plt.subplots(1, len(keys), figsize=(18, 4))
    # plot total
    for result, label in zip(results, labels):
        ax1.plot(result[keys[0]], label=label)
    ax1.set_title(keys[0])
    ax1.legend()

    # plot utility
    for result, label in zip(results, labels):
        ax2.plot(result[keys[1]], label=label)
    ax2.set_title(keys[1])
    ax2.legend()

    # plot fairness
    for result, label in zip(results, labels):
        ax3.plot(result[keys[2]], label=label)
    ax3.set_title(keys[2])
    ax3.legend()

    if save_path:
        fig.savefig(save_path + f"/comparison_subplots.png")
    if show:
        fig.show()

    plt.close()
