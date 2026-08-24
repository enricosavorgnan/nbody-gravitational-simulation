"""
Cleaning procedures
"""
import argparse
import datetime
import pandas as pd
import numpy as np
import os
import yaml

from analysis.vis import *


def info(file_path, name_info, runs) -> pd.DataFrame:
    """
    Extract the information name_info from file_path.

    Information should follow the common scheme as build by
    the profiler, meaning that there is a line ``name_info``
    and subsequently one number per line.
    The first empty line met stops the extraction.
    Extraction is then launched a number of runs ``runs``,
    always starting from where the first extraction stopped.
    This because each file contains results for several runs
    with identical setup.

    Params
    ------
    file_path : str
        Path to the .TXT file containing information to extract.
    name_info : str
        Exact name of the information to extract.
    runs : int
        Number of runs to extract.

    Returns
    -------
    data : pd.DataFrame
        A data-frame containing the extracted information, with
        shape (runs, steps), where ``steps`` is the number of
        steps in the run.
    """

    data = []
    with open(file_path, "r") as file:
        for _ in range(runs):
            run_data = []
            found = False
            for line in file:
                if line.strip() == name_info.strip():
                    found = True
                    break
            if not found:
                break

            for line in file:
                stripped = line.strip()
                if not stripped:
                    break
                run_data.append(float(stripped))

            data.append(run_data)

    return pd.DataFrame(data)




def process(data_raw : list[list[pd.DataFrame]], verbose : bool = True):
    """
    Convert the raw data into ready-to-plot data.
    If ``verbose`` is True prints also the statistics
    related to the data.

    Params
    ------
    ``data_raw``: list[list[pd.DataFrame]]
        Contains a list of lists (one for each processed file)
        of dataframes, one for each information processed
    ``verbose``: bool, default True

    Returns
    -------
    data_processed : dict
        {}
    """
    n_methods = len(data_raw)
    n_infos = len(data_raw[0])

    data_processed = {}
    for info in range(n_infos):
        for method in range(n_methods):
            df = data_raw[method][info]
            mean = df.mean(axis=0)
            trimmed_mean = df.apply(lambda x: x[x.between(x.quantile(0.25), x.quantile(0.75))].mean(), axis=0)
            std = df.std(axis=0)

            if info not in data_processed:
                data_processed[info] = {}
            data_processed[info][method] = {
                "mean": mean,
                "trimmed_mean": trimmed_mean,
                "std": std
            }

            if verbose:
                print(f"----------------------------")
                print(f"Info: {info}, Method: {method}")
                print(f"Mean: {mean.values}")
                print(f"Trimmed Mean: {trimmed_mean.values}")
                print(f"Std: {std.values}")
                print("\n")

    return data_processed


def save(figs, save_path):
    """
    Save the figures and data to disk.
    The figures are saved as .png files
    """
    # Create a timestamped directory to save the figures
    time = datetime.datetime.now().strftime("%m-%d-%H-%M")
    path = os.path.join(save_path, time)

    # Check whether the directory exists, if not create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the figures
    for i, fig in enumerate(figs):
        fig.savefig(os.path.join(path, f"figure_{i}.png"))


def main(config):

    files = config["files"]
    runs = config["runs"]
    names_info = config["names_info"]
    names_method = config["names_method"]

    data_raw = []
    for file in files:
        dfs = []
        for name in names_info:
            dfs.append(info(file_path=file, name_info=name, runs=runs))
        data_raw.append(dfs)
    data_processed = process(data_raw)

    figs = plot(data=data_processed, names_info=names_info, names_method=names_method)

    save(figs, save_path=config["save-path"])












if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="Cleaning procedures")
    parser.add_argument("--config", type=str, help="Path to the config file", required=True)
    args = parser.parse_args()

    config = args.config

    with open(config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
