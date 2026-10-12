"""
Illustrative physiological transform adapted from the supplied v1.26 package.

Inputs use an imposed CMRO2 proxy scale, not measured oxygen consumption.
Absolute inputs are converted to fractional change before the algebraic
transform. Constants and expressions remain model assumptions whose full
physiological provenance requires author review. Outputs share one driver.
CBF is returned in mL/(g*min); multiply by 100 for mL/(100g*min).
The motivation is DiNuzzo et al. (2024), doi:10.1111/jnc.15839; this code has
not been established as a complete or independently validated reproduction.
"""
import numpy as np

CONSTANTS = dict(
    n=2.28,             # neurovascular coupling constant (dCBF/CBF0 per dCMRO2/CMRO2_0)
    oef0=0.42,          # baseline oxygen extraction fraction
    pH_A=7.40,          # arterial pH
    pO2_A=100.0,        # arterial pO2 (torr) -- defined for completeness, unused
    pCO2_A=40.0,        # arterial pCO2 (torr) -- defined for completeness, unused
    P50=27.0,           # hemoglobin P50 (torr)
    h=2.8,              # Hill coefficient
    BC=26.0,            # buffering capacity
    HC=0.33,            # Haldane constant
    CF=0.14,            # hemoglobin correction factor
    Rglyc=3.1,          # non-oxidative / oxidative glycolysis ratio  (VERIFY, see docs)
    cmro2_baseline=2.1,  # umol/(g*min)
    O2_A_conc=6.9,      # arterial O2 concentration (mmol/L)
    CO2_A_conc=25.6,    # arterial CO2 concentration (mmol/L)
    fCO2_plasma=0.66,   # fraction of total CO2 carried in plasma
    fVplasma=0.55,      # plasma volume fraction of blood
    pka=6.10,           # pKa of carbonic acid in blood
    alpha_CO2=0.0307,   # CO2 solubility (mmol/L/torr)
)

VARIABLES = ('CBF', 'OEF', 'ph_V', 'p_CO2_V', 'pO2_cap', 'CMRO2', 'DeltaHCO2', 'DeltaLAC')
# Variables that are exact affine functions of delta' (identical t / d / p in any paired test)
AFFINE_VARIABLES = ('CBF', 'CMRO2', 'DeltaHCO2', 'DeltaLAC')


def neurovascular_from_cmro2(values, input_mode="absolute", constants=CONSTANTS):
    """Vectorised model.  ``values``: array-like of CMRO2 inputs (any shape)."""
    c = constants
    x = np.asarray(values, dtype=float)
    if input_mode == "absolute":
        d = x / c['cmro2_baseline'] - 1.0
    elif input_mode == "fractional_legacy":
        d = x
    else:
        raise ValueError("input_mode must be 'absolute' or 'fractional_legacy'")

    oef = c['oef0'] * (1 + d) / (1 + c['n'] * d)
    cmro2 = c['cmro2_baseline'] * (1 + d)
    cbf = cmro2 / (c['O2_A_conc'] * oef)
    delta_cmro2 = c['cmro2_baseline'] * d

    deltaH_lac = 2 * c['Rglyc'] * (1 / 6) * delta_cmro2 / cbf
    deltaH_CO2 = (1 - c['HC']) * c['O2_A_conc'] * oef
    deltaH_hemo = (1 - c['HC']) * c['O2_A_conc'] * oef   # same expression as v1.0.0 (see docs)
    deltaH_solution = deltaH_CO2 + deltaH_lac - c['CF'] * deltaH_hemo
    delta_lac_corrected = cbf * deltaH_lac
    deltaH_CO2_corrected = cbf * (deltaH_CO2 - c['CF'] * deltaH_hemo)

    pH_V = c['pH_A'] + (-1 / c['BC']) * deltaH_solution
    co2_gas = (c['CO2_A_conc'] + c['fCO2_plasma'] / c['fVplasma'] * c['O2_A_conc'] * oef) \
        / (1 + 10 ** (pH_V - c['pka']))
    pCO2_V = co2_gas / c['alpha_CO2']
    pO2_cap = c['P50'] * (2 / oef - 1) ** (1 / c['h'])

    return {'CBF': cbf, 'OEF': oef, 'ph_V': pH_V, 'p_CO2_V': pCO2_V,
            'pO2_cap': pO2_cap, 'CMRO2': cmro2,
            'DeltaHCO2': deltaH_CO2_corrected, 'DeltaLAC': delta_lac_corrected}


def calculate_neurovascular_variables(EEG_CMRO2, input_mode="absolute"):
    """Dict-of-lists API used by the GUI: ``{'<electrode> trial <n>': [values]}``.

    Returns ``{variable: {key: [floats]}}`` (same shape as v1.0.0).
    """
    out = {v: {} for v in VARIABLES}
    for key, series in EEG_CMRO2.items():
        res = neurovascular_from_cmro2(np.asarray(series, dtype=float), input_mode)
        for v in VARIABLES:
            out[v][str(key)] = [float(z) for z in np.atleast_1d(res[v])]
    return out
