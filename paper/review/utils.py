import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


APPROACHES = ["B", "TC", "SA", "SATC"]
PAIRS = list(combinations(APPROACHES, 2))
sns.set(style='whitegrid')
font = {'family' : "Times New Roman",
    'size'   : 10}
plt.rc('font', **font)
plt.rcParams.update({'mathtext.default':  'regular' })

def fdr_bh(pvals):
    p, m = np.array(pvals), len(pvals)
    order = np.argsort(p)
    p_sorted = p[order]
    
    raw = (m / np.arange(1, m + 1)) * p_sorted          # pair m/k with p(k)
    padj = np.minimum.accumulate(raw[::-1])[::-1]        # backward min accumulate
    
    padj = np.clip(padj, 0, 1)
    out = np.empty(m); out[order] = padj
    return out

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def plot_seaborn(data, metric_keys, regions, ylabel, p_values, resPath):
    if "separated" in ylabel.lower():
        plt.figure(figsize=(36, 10), dpi=300)
    else:
        plt.figure(figsize=(8, 6), dpi=300)

    rows = []
    for app in APPROACHES:
        for key, region in zip(metric_keys, regions):
            for v in data[app][key]:
                rows.append({"Approach": app, "Region": region, "value": v})
    df = pd.DataFrame(rows)
    df["Approach"] = pd.Categorical(df["Approach"], categories=APPROACHES)

    sns.boxplot(data=df, x="Region", y="value", hue="Approach")

    # Significance brackets
    n_approaches = len(APPROACHES)
    width = 0.8  # seaborn's default — this was the bug (was using 0.6)
    offsets = np.linspace(-width/2 + width/(2*n_approaches),
                           width/2 - width/(2*n_approaches),
                           n_approaches)

    for ri, region in enumerate(regions):
        region_p_values = p_values[metric_keys[ri]]
        sig_pairs = [(PAIRS[k], region_p_values[PAIRS[k]][1])
                     for k in range(len(PAIRS))
                     if region_p_values[PAIRS[k]][1] < 0.05]
        if not sig_pairs:
            continue

        pos = {a: ri + offsets[i] for i, a in enumerate(APPROACHES)}

        group_vals = [data[a][metric_keys[ri]] for a in APPROACHES]

        y_max  = max(np.max(v[np.logical_not(np.isinf(v))]) for v in group_vals)
        y_step = (max(np.max(v[np.logical_not(np.isinf(v))]) for v in group_vals) -
                  min(np.min(v) for v in group_vals)) * 0.08

        for level, ((a, b), pv) in enumerate(sig_pairs):
            y = y_max + y_step * (level + 1)
            x1, x2 = pos[a], pos[b]
            plt.plot([x1, x1, x2, x2], [y - y_step*0.1, y, y, y - y_step*0.1],
                     lw=1.1, color="black")
            plt.text((x1 + x2) / 2, y, stars(pv),
                     ha="center", va="bottom", fontsize=9)

    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(resPath, "final_{}.pdf".format(ylabel)), dpi=300)  # savefig BEFORE show
    # plt.show(block=True)


def calculate_p_values(data, groups):
    p_values = {}
    for group in groups:
        p_raw, wstats = [], []
        for a, b in PAIRS:
            if (data[a][group] - data[b][group]).nonzero()[0].size == 0:  # all values are identical, skip test
                s = np.nan; p = np.nan
            else:
                s, p = wilcoxon(data[a][group], data[b][group], nan_policy='omit')
            wstats.append(s); p_raw.append(p)
        p_adj = fdr_bh(p_raw)
        p_values[group] = {PAIRS[k]: (p_raw[k], p_adj[k]) for k in range(len(PAIRS))}
    
    return p_values

def save_p_values(p_values, resPath, filename):
    with pd.ExcelWriter(os.path.join(resPath, filename)) as writer:
        for sheet_name, d in p_values.items():
            rows = [{'Pair': f"({k[0]}, {k[1]})", 'P value raw': v[0], 'P value adjusted': v[1]} for k, v in d.items()]
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)