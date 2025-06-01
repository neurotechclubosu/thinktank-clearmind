# -*- coding: utf-8 -*-
def plot_normalized_gamma_across_channels(
    EEG_Welch_Spectra,
    ElectrodeList,
    Trials,
    global_normalization=True,
    adaptive_clip_percentile=0  # e.g., set to 5 for 5th–95th percentile clipping
):
    """
    Converts EEG Welch spectra into normalized Gamma-band power estimates
    and estimates CMRO₂ from those values.

    Parameters
    ----------
    EEG_Welch_Spectra : dict
        Output from EEG_Implement_Welch containing Welch-transformed EEG data.
    ElectrodeList : list
        List of electrode names used.
    Trials : int
        Number of trials.
    global_normalization : bool
        Whether to normalize across all trials/electrodes or per-trial.
    adaptive_clip_percentile : float
        If > 0, clips gamma power to [p, 100-p] percentile range before scaling.

    Returns
    -------
    WelchPoints : dict
        Dictionary of normalized CMRO2 values per trial and electrode.
    """
    import numpy as np

    # Step 1: Collect gamma values
    all_gamma_values = []
    trial_gamma_map = {}  # Store per trial for later if per-trial normalization

    for TrialsKeep in range(Trials):
        for electrode in ElectrodeList:
            trial_key = f'{electrode} trial {TrialsKeep}'
            if trial_key not in EEG_Welch_Spectra:
                continue

            gamma_power = [
                seg['band_power']['Gamma'] for seg in EEG_Welch_Spectra[trial_key]['segments']
            ]
            trial_gamma_map[trial_key] = gamma_power
            all_gamma_values.extend(gamma_power)

    # Step 2: Determine normalization bounds
    if global_normalization:
        if adaptive_clip_percentile > 0:
            low = np.percentile(all_gamma_values, adaptive_clip_percentile)
            high = np.percentile(all_gamma_values, 100 - adaptive_clip_percentile)
        else:
            low = min(all_gamma_values)
            high = max(all_gamma_values)

    # Step 3: Normalize and convert to CMRO2
    WelchPoints = {}

    for trial_key, gamma_power in trial_gamma_map.items():
        if global_normalization:
            norm_min, norm_max = low, high
        else:
            if adaptive_clip_percentile > 0:
                norm_min = np.percentile(gamma_power, adaptive_clip_percentile)
                norm_max = np.percentile(gamma_power, 100 - adaptive_clip_percentile)
            else:
                norm_min = min(gamma_power)
                norm_max = max(gamma_power)

        # Avoid division by zero
        if norm_max - norm_min == 0:
            normalized = [2.1 for _ in gamma_power]
        else:
            normalized = [
                2.1 + ((val - norm_min) / (norm_max - norm_min)) for val in gamma_power
            ]

        WelchPoints[trial_key] = normalized

    return WelchPoints
