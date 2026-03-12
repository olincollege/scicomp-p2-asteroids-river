"""
Plot the results of a sweep.
Takes a file in data/sweeps/*all_sweep_results.csv from step 21
and generates plots of the results.
Specifically, plots 1 or 2 parameters against the result metrics:
num_families, num_non_family, v_measure, total_carrie_measure.
"""

import os
from matplotlib.ticker import FuncFormatter
import pandas as pd
import matplotlib.pyplot as plt

def main():
    sweep_dir = os.path.join("data", "sweeps")
    files = os.listdir(sweep_dir)
    print(f"Found {len(files)} files in {sweep_dir}. Looking for files ending with '_all_sweep_results.csv'...")
    possible_files = [filename for filename in files if filename.endswith("_all_sweep_results.csv")]
    
    print("Possible files to plot: ")
    for i, filename in enumerate(possible_files):
        print(f"{i}: {filename}")
    file_idx = int(input(f"Enter the number of the file to plot (0-{len(possible_files)-1}): "))
    if file_idx < 0 or file_idx >= len(possible_files):
        print("Invalid file number, exiting.")
        return
    file_to_plot = possible_files[file_idx]
    print(f"Loading data from {file_to_plot}...")
    df = pd.read_csv(os.path.join(sweep_dir, file_to_plot), skipinitialspace=True)
    print(f"Data loaded. Columns: {df.columns}")
    # find all parameter columns
    metric_names = ["num_families", "num_non_family", "v_measure", "total_carrie_measure"]
    param_names = [col for col in df.columns if col not in metric_names]
    param_1 = None
    param_2 = None
    print("Parameter columns found:")
    for i, col in enumerate(param_names):
        print(f"{i}: {col}")
    param_1_idx = int(input(f"Enter the number of the first parameter to plot (0-{len(param_names)-1}): "))
    if param_1_idx < 0 or param_1_idx >= len(param_names):
        print("Invalid parameter number, exiting.")
        return
    param_1 = param_names[param_1_idx]
    param_2_idx = int(input(f"Enter the number of the second parameter to plot (or -1 to skip): "))
    if param_2_idx >= 0:
        if param_2_idx < 0 or param_2_idx >= len(param_names):
            print("Invalid parameter number, exiting.")
            return
        param_2 = param_names[param_2_idx]
    print(f"Plotting results with param_1={param_1} (min={df[param_1].min()}, max={df[param_1].max()}) and param_2={param_2} (min={df[param_2].min()}, max={df[param_2].max()})...")
    # plot the results in a 2x2 grid of subplots, one for each metric
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Sweep results from {file_to_plot}")
    # if there are two parameters, the plot should be a heatmap with param_1 on the x-axis and param_2 on the y-axis, and the metric as the color
    # if there is one parameter, the plot should be a line plot with param_1 on the x-axis and the metric on the y-axis
    
    for i, metric in enumerate(metric_names):
        ax = axs[i // 2, i % 2]
        if param_2:
            pivot = df.pivot_table(index=param_2, columns=param_1, values=metric)
            im = ax.imshow(pivot, aspect='auto', origin='lower', extent=(df[param_1].min(), df[param_1].max(), df[param_2].min(), df[param_2].max()), cmap='viridis')
            ax.set_xlabel(param_1)
            ax.set_ylabel(param_2)
            ax.set_title(metric)
            fig.colorbar(im, ax=ax)
        else:
            ax.plot(df[param_1], df[metric])
            ax.set_xlabel(param_1)
            ax.set_ylabel(metric)
            ax.set_title(metric)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
