import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import numpy as np
import os

X_COLUMN = 'Components'

Y_COLUMNS_TO_PLOT = ['Weight Norm']
#Y_COLUMNS_TO_PLOT = ['Test 0-1 Loss','Train 0-1 Loss']
#Y_COLUMNS_TO_PLOT = [ 'Test MSE']

FILE_DIRECTORY = '.'

FILE_PATTERN = 'cifar_nsamp=*.csv' 

def create_plots_per_gamma(directory, pattern, x_col, y_cols):
    """
    Finds CSVs, groups them by gamma, averages over seeds, and
    creates a separate plot for each gamma, plotting all
    columns specified in y_cols.
    """

    search_path = os.path.join(directory, pattern)
    file_list = glob.glob(search_path)

    if not file_list:
        print(f"No files found matching pattern: {search_path}")
        print("Please upload your CSV files and try again.")
        return

    regex_pattern = re.compile(r"nlvl=(.*?)_gamma=(.*?)_seed=.*?.csv")

    data_by_gamma = {}

    print(f"Found {len(file_list)} files. Processing...")

    for f_path in file_list:
        filename = os.path.basename(f_path)
        match = regex_pattern.search(filename)

        if not match:
            print(f"Warning: Skipping file (pattern mismatch): {filename}")
            continue

        nlvl = match.group(1)
        gamma_str = match.group(2)

        try:

            if gamma_str.startswith('d'):
                gamma_str_cleaned = gamma_str[1:]
            else:
                gamma_str_cleaned = gamma_str

            gamma_str_cleaned = gamma_str_cleaned.replace('d', '.')

            gamma_float = float(gamma_str_cleaned)

        except ValueError:
            print(f"Warning: Skipping file (could not parse gamma '{gamma_str}'): {filename}")
            continue

        group_key = (nlvl, gamma_float)

        try:
            df = pd.read_csv(f_path)

            required_cols = [x_col] + y_cols
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"Error: File {filename} is missing required columns: {missing_cols}")
                print(f"Available columns are: {list(df.columns)}")
                print("Please update X_COLUMN and Y_COLUMNS_TO_PLOT variables.")
                return

            if group_key not in data_by_gamma:
                data_by_gamma[group_key] = []
            data_by_gamma[group_key].append(df)

        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty file: {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not data_by_gamma:
        print("No data was successfully processed.")
        return

    print(f"\nSuccessfully grouped data into {len(data_by_gamma)} (nlvl, gamma) combinations.")

    plot_filenames = []

    for (nlvl, gamma), df_list in data_by_gamma.items():
        if not df_list:
            continue

        print(f"\nProcessing group: nlvl={nlvl}, gamma={gamma} ({len(df_list)} seeds)")

        try:

            combined_df = pd.concat(df_list)
        except Exception as e:
            print(f"Error concatenating dataframes for gamma={gamma}: {e}")
            continue

        grouped = combined_df.groupby(x_col)
        mean_df = grouped[y_cols].mean()
        std_df = grouped[y_cols].std()

        mean_df = mean_df.add_suffix('_mean')
        std_df = std_df.add_suffix('_std')

        plot_df = mean_df.join(std_df).reset_index()

        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        for i, col_name in enumerate(y_cols):
            mean_col = f"{col_name}_mean"
            std_col = f"{col_name}_std"

            ax.plot(plot_df[x_col], plot_df[mean_col], 
                    label=f'{col_name} (Mean)', 
                    marker='o', markersize=4)

            ax.fill_between(
                plot_df[x_col],
                plot_df[mean_col] - plot_df[std_col],
                plot_df[mean_col] + plot_df[std_col],
                alpha=0.2,
                label=f'{col_name} (+/- 1 Std Dev)'
            )

        gamma_str_filename = str(gamma).replace('.', 'p')
        plt.title(f'Weight vs. Components (Gamma = 0.002, nlvl = {nlvl})')
        plt.xlabel(x_col)
        plt.ylabel('Weight')
        plt.yscale('log')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.grid(True, linestyle='--', alpha=0.6)

        plt.show()

    print("\n--- All plots saved ---")
    for fname in plot_filenames:
        print(fname)

if __name__ == "__main__":
    create_plots_per_gamma(
        FILE_DIRECTORY, 
        FILE_PATTERN, 
        X_COLUMN, 
        Y_COLUMNS_TO_PLOT,
    )
