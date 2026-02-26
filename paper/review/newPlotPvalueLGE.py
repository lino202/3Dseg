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
    if "nnUNet" in args.filePath:
        for lbl in APPROACHES:
            with open(os.path.join(args.filePath, 'results_{}_originalSpace_metrics'.format(lbl), "results.pickle"), 'rb') as f:
                data[lbl] = pickle.load(f)

        for lbl in APPROACHES:
            with open(os.path.join(args.filePath, 'results_{}_originalSpace_metrics'.format(lbl), "results_separatedBE.pickle"), 'rb') as f:
                data_be_separated[lbl] = pickle.load(f)
    else:
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
    groups = ['gdsc_lv', 'gdsc_myo', 'gdsc_mi', 'hd_lv', 'hd_myo', 'hd_mi', 'assd_lv', 'assd_myo', 'assd_mi', 'be', 'ts']
    p_values = calculate_p_values(data, groups)

    
    groups_be_separated = ['be_lv_0', 'be_lv_1', 'be_lv_2', 'be_myo_0', 'be_myo_1', 'be_myo_2', 'be_mi_0', 'be_mi_1', 'be_mi_2', 
              'be_lvmyo_0', 'be_lvmyo_1', 'be_lvmyo_2', 'be_lvmi_0', 'be_lvmi_1', 'be_lvmi_2', 'be_myomi_0', 'be_myomi_1', 'be_myomi_2']
    p_values_be_separated = calculate_p_values(data_be_separated, groups_be_separated)



    # ── Plotting ────────────────────────────────────────────────────
    plot_seaborn(data, ["gdsc_lv", "gdsc_myo", "gdsc_mi"],
                        ["LV", "MYO", "MI"], "DSC (unitless)", p_values, args.filePath)
    plot_seaborn(data, ["hd_lv",   "hd_myo",   "hd_mi"],
                        ["LV", "MYO", "MI"], "HD (mm)", p_values, args.filePath)
    plot_seaborn(data, ["assd_lv", "assd_myo", "assd_mi"],
                        ["LV", "MYO", "MI"], "ASSD (mm)", p_values, args.filePath)
    
    plot_seaborn(data, ["be"], ["BE"], "BE (unitless)", p_values, args.filePath)

    plot_seaborn(data_be_separated, 
                 ['be_lv_0', 'be_lv_1', 'be_lv_2', 'be_myo_0', 'be_myo_1', 'be_myo_2', 'be_mi_0', 'be_mi_1', 'be_mi_2', 
              'be_lvmyo_0', 'be_lvmyo_1', 'be_lvmyo_2', 'be_lvmi_0', 'be_lvmi_1', 'be_lvmi_2', 'be_myomi_0', 'be_myomi_1', 'be_myomi_2'], 
              ["lv_B0", "lv_B1", "lv_B2", "myo_B0", "myo_B1", "myo_B2", "mi_B0", "mi_B1", "mi_B2", 
              "lvmyo_B0", "lvmyo_B1", "lvmyo_B2", "lvmi_B0", "lvmi_B1", "lvmi_B2", "myomi_B0", "myomi_B1", "myomi_B2"], "BE separated (unitless)", p_values_be_separated, args.filePath)


    # Save p values
    save_p_values(p_values, args.filePath, 'p_values.xlsx')
    save_p_values(p_values_be_separated, args.filePath, 'p_values_be_separated.xlsx')
    
    print("Saved to p_values.xlsx and p_values_be_separated.xlsx")

if __name__ == '__main__':
    main()