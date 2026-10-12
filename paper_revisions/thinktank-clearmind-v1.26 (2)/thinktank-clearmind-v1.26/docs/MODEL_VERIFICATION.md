# Neurovascular model constants: verification status

Steps2 / Reviewer #4 point 2 asked that "all neurovascular equations,
constants, and units" be verified. This is the audit. Model:
`src/eeg/EEG_NeurovascularVariables.CONSTANTS`, an arteriovenous flux-balance
implementation following the homeostasis model of DiNuzzo et al. (2024,
*J. Neurochem.* 168:632-662, doi:10.1111/jnc.15839).

| constant | value | status | note |
|---|---|---|---|
| `n` (coupling constant) | 2.28 | Plausible, within published range | DiNuzzo et al. report n in the 2-3 range across conditions/subjects; unchanged from v1.0.0; **not independently re-derived for this revision** — flagged for the authors to cite the exact source value/table used. |
| `oef0` (baseline OEF) | 0.42 | Standard textbook value | Matches widely-cited resting OEF (~0.40-0.44); unchanged from v1.0.0. |
| `pH_A` | 7.40 | Standard | Normal arterial pH. |
| `P50` | 27.0 torr | Standard | Human hemoglobin P50 at physiological pH/temperature. |
| `h` (Hill coefficient) | 2.8 | Standard | Typical hemoglobin cooperativity value (2.7-2.9 commonly cited). |
| `BC` (buffering capacity) | 26.0 | Plausible | Consistent with brain-tissue non-bicarbonate buffering capacity estimates; unchanged from v1.0.0, **not independently re-derived**. |
| `HC` (Haldane constant) | 0.33 | Plausible | Order-of-magnitude consistent with Haldane-effect literature values; unchanged from v1.0.0, **not independently re-derived**. |
| `CF` (hemoglobin correction) | 0.14 | **Not independently verified** | Small correction term; no external source located during this revision. Its effect on OEF and CMRO2 (the manuscript's primary/affine outputs) is exactly zero — it enters only the ΔHCO2/ΔLAC intermediate terms — so it does not affect the montage-mean CMRO2 statistics reported. Flagged for the authors to source or remove. |
| `Rglyc` (glycolysis ratio) | 3.1 | **Not independently verified** | Same status as `CF`; affects `DeltaLAC` only, which is one of the seven variables shown (Methods) to be non-independent of CMRO2/the other affine outputs. Does not affect CMRO2, CBF, OEF, or the statistical conclusions. |
| `cmro2_baseline` | 2.1 umol/(g*min) | Standard | Commonly cited resting grey-matter CMRO2 (~1.5-2.2 umol/g/min range depending on tissue/species); this is also the value the application's gamma-normalization stage independently uses as its scale floor, so both stages are now mutually consistent (this consistency was **not** true in v1.0.0 due to the units defect, C-5 in `CORRECTIONS.md`). |
| `O2_A_conc`, `CO2_A_conc`, `fCO2_plasma`, `fVplasma`, `pka`, `alpha_CO2` | as listed | Standard blood-gas physiology values | Unchanged from v1.0.0; consistent with standard physiology references (e.g. arterial O2 content ~6.5-7 mmol/L at normal hemoglobin/saturation, plasma CO2 solubility ~0.03 mmol/L/torr, carbonic acid pKa ~6.1). |

## What this revision changed vs. did not change

* **Fixed**: the units mismatch between the gamma-normalization stage's output
  (absolute CMRO2) and the model's expected input (fractional change), C-5 in
  `CORRECTIONS.md`. This is a correctness fix, not a constants change, and is
  the only change that alters any reported number.
* **Not changed**: none of the fourteen constants above were altered from
  their v1.0.0 values. Three (`CF`, `Rglyc`, and, to a lesser extent, `n`,
  `BC`, `HC`) could not be traced to a specific cited source within this
  revision's scope and are flagged rather than silently kept; none of them
  affects the CMRO2, CBF or OEF statistics that carry the manuscript's
  quantitative claims (see `AFFINE_VARIABLES` in the module and the
  dependence analysis in `CORRECTIONS.md`).

## Recommended follow-up before final submission

1. Cite the exact page/table of DiNuzzo et al. (2024) — or an earlier source,
   if `n`, `BC`, `HC`, `CF`, `Rglyc` were taken from a different paper — for
   every constant in the table above, in the Methods.
2. If `CF` and `Rglyc` cannot be sourced, consider reporting ΔHCO2 and ΔLAC
   as "illustrative, not independently verified" in the affected
   table/figure captions, consistent with how the dependence limitation is
   already phrased.
