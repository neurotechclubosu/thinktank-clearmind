import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

def compute_channel_means(data):
    """
    Given a dict mapping metric → { "Channel trial N": [32 values], … },
    returns a dict: metric → { channel: mean_over_all_trials }.
    Steps:
      1. For each "Channel trial N", compute mean of its 32 timepoints.
      2. Group by channel and average those per-trial means.
    """
    channel_means = {}
    for metric, trials in data.items():
        per_channel_trials = {}
        for trial_name, values in trials.items():
            channel = trial_name.split(" trial")[0]
            trial_mean = np.mean(values)
            per_channel_trials.setdefault(channel, []).append(trial_mean)
        final_means = {ch: np.mean(vs) for ch, vs in per_channel_trials.items()}
        channel_means[metric] = final_means
    return channel_means

# ——— Load JSONs for both conditions ———
with open(r"neurovascular_data\Wu_Tang_Clan_s_raw_mg1_neurovascular.json", "r") as f:
    data_with = json.load(f)

with open(r"neurovascular_data\Wu-Tang_Clan's_raw_unfiltered_narratives_mg2_neurovascular.json", "r") as f:
    data_without = json.load(f)

# ——— Compute per-channel averages per metric ———
means_with = compute_channel_means(data_with)
means_without = compute_channel_means(data_without)

# ——— Run paired t-tests, Wilcoxon, and Cohen's d ———
results = {}

for metric in means_with:
    channels = sorted(means_with[metric].keys())
    arr_with = np.array([means_with[metric][ch] for ch in channels])
    arr_without = np.array([means_without[metric][ch] for ch in channels])

    # Paired t-test
    t_stat, p_val_t = ttest_rel(arr_with, arr_without)

    # Wilcoxon signed-rank (nonparametric)
    try:
        w_stat, p_val_w = wilcoxon(arr_with, arr_without)
    except ValueError:
        p_val_w = np.nan

    # Cohen’s d for paired samples
    diff = arr_with - arr_without
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff != 0 else np.nan

    results[metric] = {
        "t_statistic": float(t_stat),
        "p_value_ttest": float(p_val_t),
        "p_value_wilcoxon": float(p_val_w),
        "cohens_d": float(cohens_d),
        "n_pairs": len(channels)
    }

# Convert to DataFrame
df_results = (
    pd.DataFrame(results)
      .T
      .reset_index()
      .rename(columns={"index": "Metric"})
)

print(df_results.to_string(index=False))
