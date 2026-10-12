# Kara One independent four-word baseline evaluation

MM05 was a format-only pilot, excluded.13 new people, four words: gnaw, knew, pat, pot; chance25%. All recordings are public secondary data on D:. This is a different task from FEIS, not exact replication and not full-sentence reconstruction.

| Model | Condition | Seed | People | Balanced accuracy | Participant95% interval |
|---|---|---:|---:|---:|---|
| amplitude_nuisance | primary | 17 | 13 | 23.65% | 17.88-29.90% |
| covariance | eye_regression | 17 | 13 | 26.06% | 19.81-33.17% |
| covariance | global_chronology | 17 | 13 | 22.82% | 14.68-31.12% |
| covariance | primary | 17 | 13 | 28.85% | 23.56-35.58% |
| covariance | strict | 17 | 13 | 29.17% | 23.56-36.06% |
| eegnet_style | eye_regression | 17 | 13 | 27.69% | 22.60-32.98% |
| eegnet_style | global_chronology | 17 | 13 | 21.19% | 14.34-27.92% |
| eegnet_style | primary | 17 | 13 | 25.10% | 20.67-29.42% |
| eegnet_style | strict | 17 | 13 | 24.78% | 20.29-29.58% |
| eegnet_style | primary | 29 | 13 | 22.60% | 16.83-27.88% |
| eegnet_style | primary | 43 | 13 | 25.38% | 20.67-30.67% |
| eye_only | primary | 17 | 13 | 30.96% | 24.04-38.17% |

Primary decisions use seed17 and Holm correction across two families; other seeds/controls/sensitivities are secondary. The primary split is later within each class and can interleave across classes; the global chronological sensitivity has genuinely later real test epochs and its own training-only selection.

covariance: conditional permutation p=0.092, Holm p=0.184; provisional>=30%/CI-above25% gate **failed**.

eegnet_style: conditional permutation p=0.516, Holm p=0.516; provisional>=30%/CI-above25% gate **failed**.

Conditional permutations assume exchangeable labels within the held-out participant portion; they do not establish neural specificity. Eye-only and raw-amplitude controls, training-fitted eye regression and chronology sensitivities remain visible. A negative eye-only result cannot rule out other artifacts; no direct muscle channel is verified. The channel called EMG is a task-colour sensor and is excluded from predictors.

Acquisition/label checks were conducted before classifier scores, including colour-sensor corroboration where repeats made text alignment ambiguous. Supplied imagery intervals are task periods, not measured internal speech onsets. Source/calibration assumptions and omissions are in participant audits.

Independent verification recovered 156 scores from 2540 real predictions, checked native labels, trial/fold isolation, cached EEG and eye-regression receipts, and source snapshots.

No results support calibrated CMRO2 or continuous sentence biology. No paper, collaborator message, journal submission or public deposit was changed by this experiment.
