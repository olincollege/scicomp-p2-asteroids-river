"""
Parallelized parameter sweeps generate a lot of data/sweeps/<alg>-<dset><params>.csv files.
This script concatenates them into a single file for easier analysis.
"""

import os
import pandas as pd

from common.classifiers import choose_classifier_interactive
from common.datasets import choose_dataset_interactive, load_dataset
from common.inputs import build_arg_parser

def main():
    sweep_dir = os.path.join("data", "sweeps")
    parser = build_arg_parser("Sweep over a range of parameter values for a chosen clustering algorithm.")
    args, _ = parser.parse_known_args()

    # dataset
    dataset_name = args.dataset if args.dataset else choose_dataset_interactive()

    # classifier
    classifier_name = args.classifier if args.classifier else choose_classifier_interactive()

    output_file = f"{classifier_name}_{dataset_name}_all_sweep_results.csv"
    header_written = False
    files_written = 0

    files = os.listdir(sweep_dir)
    print(f"Found {len(files)} files in {sweep_dir}. Processing files for classifier '{classifier_name}' and dataset '{dataset_name}'...")
    for filename in files:
        if filename.startswith(f"{classifier_name}_{dataset_name}"):
            file_path = os.path.join(sweep_dir, filename)
            df = pd.read_csv(file_path)
            df.to_csv(os.path.join(sweep_dir, output_file), mode='a', header=not header_written, index=False)
            header_written = True
            files_written += 1
    print(f"Concatenated {files_written} files into {output_file}.")

if __name__ == "__main__":
    main()
