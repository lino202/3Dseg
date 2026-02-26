import numpy as np
import argparse
import os
import pickle
from utils import APPROACHES, plot_seaborn, calculate_p_values, save_p_values



def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--filePath',  required=True, type=str)
    args = parser.parse_args()
    
    data = {}
    data_be_separated = {}

    # Collect data for each method and metric
    with open(os.path.join(args.filePath, "results.pickle"), 'rb') as f:
        data = pickle.load(f)

    with open(os.path.join(args.filePath, "results_separatedBE.pickle"), 'rb') as f:
        data_be_separated = pickle.load(f)

    # We eliminate inf values (if any) to avoid plotting and stats issues.
    # We also print a warning with the indices of the inf values for transparency.
    for approach in data.keys():
        for key in data[approach].keys():
            if np.isinf(data[approach][key]).any():
                print(f"Warning: Found inf values in approach {approach}, class {key} and index {np.where(np.isinf(data[approach][key]))[0]} replacing with NaN for plotting and stats.")
                data[approach][key][np.isinf(data[approach][key])] = np.nan        


    # ── Significance ─────────────────────────────────────────────────────
    groups = ['gdsc_lv', 'gdsc_myo', 'gdsc_rv', 'hd_lv', 'hd_myo', 'hd_rv', 'assd_lv', 'assd_myo', 'assd_rv', 'be', 'ts']
    p_values = calculate_p_values(data, groups)

    
    groups_be_separated = ['be_lv_0', 'be_lv_1', 'be_lv_2', 'be_myo_0', 'be_myo_1', 'be_myo_2', 'be_rv_0', 'be_rv_1', 'be_rv_2', 
              'be_lvmyo_0', 'be_lvmyo_1', 'be_lvmyo_2', 'be_lvrv_0', 'be_lvrv_1', 'be_lvrv_2', 'be_myorv_0', 'be_myorv_1', 'be_myorv_2']
    p_values_be_separated = calculate_p_values(data_be_separated, groups_be_separated)



    # ── Plotting ────────────────────────────────────────────────────
    plot_seaborn(data, ["gdsc_lv", "gdsc_myo", "gdsc_rv"],
                        ["LV", "MYO", "RV"], "DSC (unitless)", p_values, args.filePath)
    plot_seaborn(data, ["hd_lv",   "hd_myo",   "hd_rv"],
                        ["LV", "MYO", "RV"], "HD (mm)", p_values, args.filePath)
    plot_seaborn(data, ["assd_lv", "assd_myo", "assd_rv"],
                        ["LV", "MYO", "RV"], "ASSD (mm)", p_values, args.filePath)
    
    plot_seaborn(data, ["be"], ["BE"], "BE (unitless)", p_values, args.filePath)

    plot_seaborn(data_be_separated, 
                 ['be_lv_0', 'be_lv_1', 'be_lv_2', 'be_myo_0', 'be_myo_1', 'be_myo_2', 'be_rv_0', 'be_rv_1', 'be_rv_2', 
              'be_lvmyo_0', 'be_lvmyo_1', 'be_lvmyo_2', 'be_lvrv_0', 'be_lvrv_1', 'be_lvrv_2', 'be_myorv_0', 'be_myorv_1', 'be_myorv_2'], 
              ["lv_B0", "lv_B1", "lv_B2", "myo_B0", "myo_B1", "myo_B2", "rv_B0", "rv_B1", "rv_B2", 
              "lvmyo_B0", "lvmyo_B1", "lvmyo_B2", "lvrv_B0", "lvrv_B1", "lvrv_B2", "myorv_B0", "myorv_B1", "myorv_B2"], "BE separated (unitless)", p_values_be_separated, args.filePath)


    # Save p values
    save_p_values(p_values, args.filePath, 'p_values.xlsx')
    save_p_values(p_values_be_separated, args.filePath, 'p_values_be_separated.xlsx')
    
    print("Saved to p_values.xlsx and p_values_be_separated.xlsx")

if __name__ == '__main__':
    main()