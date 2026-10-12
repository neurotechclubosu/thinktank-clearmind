"""
CMRO2 -> neurovascular variables (arteriovenous flux-balance model).

The equations follow the homeostasis model of DiNuzzo et al. (2024, J Neurochem
168:632-662, doi:10.1111/jnc.15839) as implemented in v1.0.0; constants are
collected in ``CONSTANTS`` so they can be audited in one place (see
docs/MODEL_VERIFICATION.md for the verification status of each one).

v1.1.0 change -- input units (docs/CORRECTIONS.md, C-5)
-------------------------------------------------------
The model's driver is the *fractional* CMRO2 increase
    delta' = (CMRO2 - CMRO2_0) / CMRO2_0,
with neurovascular coupling  dCBF/CBF_0 = n * delta'  (n ~= 2.3).  In v1.0.0 the
normalized-gamma stage returned an *absolute* CMRO2 on a 2.1-3.1 umol/g/min scale
and that array was passed straight in as ``delta'``, i.e. CMRO2 was treated as a
210-310 % increase over baseline (output CMRO2 6.5-8.6 umol/g/min; CBF
~420-585 mL/100 g/min).  ``input_mode="absolute"`` (default) converts first:
delta' = CMRO2_abs / 2.1 - 1, so a window at baseline maps to exactly
OEF = 0.42, CBF_0, CMRO2 = 2.1.  ``input_mode="fractional_legacy"`` reproduces
the v1.0.0 behaviour for regression testing only.

All eight outputs are deterministic functions of the single scalar delta'
(four are exact affine functions of it); they are not independent observations.
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
