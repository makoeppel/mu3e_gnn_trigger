import numpy as np
import matplotlib.pyplot as plt


def plot_latent_variable_distributions(
    bg_log_var, sig_log_var, **kwargs
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Plots the distributions of latent variables for background and signal data.

    Args:
        bg_log_var (np.ndarray): Background log variance array.
        sig_log_var (np.ndarray): Signal log variance array.
        kwargs: Additional keyword arguments for customization, such as:
            - x_label (str): Label for the x-axis.
            - n_bins (int): Number of bins for the histogram.
            - y_label (str): Label for the y-axis.
            - signal_label (str): Label for the signal data in the legend.
            - background_label (str): Label for the background data in the legend.

    Returns:
        fig: The matplotlib figure object.
        ax_array: The array of axes used for plotting.
    """
    if bg_log_var.ndim == 2 or sig_log_var.ndim == 2:
        if bg_log_var.shape[1] != sig_log_var.shape[1]:
            raise ValueError(
                "Background and signal log variance arrays must have the same number of variables."
            )

        num_variables = bg_log_var.shape[1]
    elif bg_log_var.ndim == 1 and sig_log_var.ndim == 1:
        num_variables = 1
    else:
        raise ValueError(
            f"Input arrays must be 2D or 1D. Got shapes {bg_log_var.shape} and {sig_log_var.shape}."
        )

    num_cols = np.ceil(np.sqrt(num_variables)).astype(int)

    num_rows = (num_variables + num_cols - 1) // num_cols  # Calculate rows needed
    fig, ax_array = plt.subplots(
        figsize=(num_cols * 3.5, num_rows * 2.5), nrows=num_rows, ncols=num_cols
    )
    ax_array = ax_array.flatten()  # Flatten the axes array for easier indexing

    x_label = kwargs.get("x_label", "$\\sigma$")
    n_bins = kwargs.get("n_bins", 30)
    y_label = kwargs.get("y_label", "Density")
    signal_label = kwargs.get("signal_label", "Signal")
    background_label = kwargs.get("background_label", "Background")

    for i in range(num_variables):
        ax = ax_array[i]
        bg_ad_score = bg_log_var[:, i]
        signal_ad_score = sig_log_var[:, i]
        bins = np.linspace(
            min(np.min(bg_ad_score), np.min(signal_ad_score)),
            max(np.max(bg_ad_score), np.max(signal_ad_score)),
            n_bins,
        )
        ax.hist(
            bg_ad_score,
            bins=bins,
            alpha=0.5,
            label=background_label,
            color="blue",
            density=True,
        )
        ax.hist(
            signal_ad_score,
            bins=bins,
            alpha=0.5,
            label=signal_label,
            color="red",
            density=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{x_label} {i+1}")
        ax.set_ylabel(y_label)

    # Hide unused subplots
    for j in range(num_variables, len(ax_array)):
        fig.delaxes(ax_array[j])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout()
    return fig, ax_array.reshape((num_rows, num_cols))
