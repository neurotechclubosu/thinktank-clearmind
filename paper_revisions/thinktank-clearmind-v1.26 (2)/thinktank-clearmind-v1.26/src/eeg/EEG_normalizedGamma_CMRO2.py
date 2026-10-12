# -*- coding: utf-8 -*-
def plot_normalized_gamma_across_channels(
    EEG_Welch_Spectra,
    ElectrodeList,
    Trials,
    global_normalization=True,
    adaptive_clip_percentile=0,  # e.g., set to 5 for 5th–95th percentile clipping
    reference_bounds=None        # (low, high) gamma-power bounds shared across recordings
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
    reference_bounds : (float, float) or None
        v1.1.0.  If given, the gamma->CMRO2 map uses these fixed bounds instead
        of the recording's own min/max, so that two recordings are placed on one
        common scale (values outside the bounds are clipped).  Default None keeps
        the published behaviour, in which every recording is min-max scaled
        against itself and absolute levels are therefore not comparable across
        recordings.

    Returns
    -------
    WelchPoints : dict
        Dictionary of normalized CMRO2 values per trial and electrode.
    """
    import numpy as np

    # Step 1: Collect gamma values
    all_gamma_values = []
    trial_gamma_map = {}  # Store per trial for later if per-trial normalization

    # Trial keys are 1-based ('... trial 1' .. '... trial N'); range(Trials)
    # looked for a nonexistent trial 0 and dropped the last trial.
    for TrialsKeep in range(1, Trials + 1):
        for electrode in ElectrodeList:
            trial_key = f'{electrode} trial {TrialsKeep}'
            if trial_key not in EEG_Welch_Spectra:
                continue

            gamma_power = [
                seg['band_power']['Gamma'] for seg in EEG_Welch_Spectra[trial_key]['segments']
            ]
            trial_gamma_map[trial_key] = gamma_power
            all_gamma_values.extend(gamma_power)

    if not all_gamma_values:
        raise ValueError(
            "No gamma values found (no complete trials for the requested electrodes); "
            "the utterance is too short for the analysis stage.")

    # Step 2: Determine normalization bounds
    if reference_bounds is not None:
        low, high = reference_bounds
        global_normalization = True
    elif global_normalization:
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
                2.1 + min(1.0, max(0.0, (val - norm_min) / (norm_max - norm_min)))
                if reference_bounds is not None else
                2.1 + ((val - norm_min) / (norm_max - norm_min))
                for val in gamma_power
            ]

        WelchPoints[trial_key] = normalized

    return WelchPoints
