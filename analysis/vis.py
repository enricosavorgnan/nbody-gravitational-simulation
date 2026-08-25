"""
Plots
"""
import pandas
import matplotlib.pyplot as plt


def plot(data : dict, names_info=None, names_method=None):
    """
    Receives a data dict and plots the data in
    different figures, one for each information processed.

    Params
    ------
    ``data``: dict
        Contains the data to plot, in the format returned by the process function.
    ``names_info``: list[str], optional
        List of names to use for the figures' titles.
        If None, uses the keys of the data dict.
    ``names_method`: list[str], optional
        List of names to use as the figures' legend.
        If None, uses the keys of the data dict`

    Returns
    -------
    figs : list[matplotlib.figure.Figure]
        List of figures generated, one for each information processed.
    """
    figs = []
    for i, (info, methods) in enumerate(data.items()):
        fig, ax = plt.subplots(dpi=600)
        for j, (method, stats) in enumerate(methods.items()):
            ax.plot(stats['trimmed_mean'], label=f'{names_method[j]}', linewidth=1., linestyle="dashed")
        ax.set_title(f"{names_info[i]} Trimmed Means")
        ax.set_xlabel('Steps')
        ax.set_ylabel('Time(s)')
        ax.legend()
        figs.append(fig)
    return figs

