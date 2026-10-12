# Stronger FEIS baseline follow-up

Exploratory follow-up: the first FEIS test results were already inspected. Native later-block real-trial holdout; no sentence/metabolism or cross-session validation.

| Phase | Model | People | Balanced accuracy | Participant95% interval |
|---|---|---:|---:|---|
| resting | covariance | 21 | 5.85% | 4.56-7.14% |
| resting | eegnet_style | 21 | 7.74% | 6.55-8.83% |
| resting | mean_power_lda | 21 | 6.05% | 4.56-7.74% |
| resting | nuisance | 21 | 6.25% | 4.96-7.64% |
| speaking | covariance | 1 | 6.25% | one-person diagnostic |
| speaking | eegnet_style | 1 | 4.17% | one-person diagnostic |
| speaking | mean_power_lda | 1 | 8.33% | one-person diagnostic |
| speaking | nuisance | 1 | 8.33% | one-person diagnostic |
| stimuli | covariance | 1 | 6.25% | one-person diagnostic |
| stimuli | eegnet_style | 1 | 4.17% | one-person diagnostic |
| stimuli | mean_power_lda | 1 | 8.33% | one-person diagnostic |
| stimuli | nuisance | 1 | 8.33% | one-person diagnostic |
| thinking | covariance | 21 | 5.56% | 4.07-7.14% |
| thinking | eegnet_style | 21 | 6.05% | 4.56-7.54% |
| thinking | mean_power_lda | 21 | 6.75% | 5.36-8.13% |
| thinking | nuisance | 21 | 7.04% | 5.16-9.23% |

Nominal16-class chance6.25%. CNN primary seed17; seeds29/43 are stability sensitivities. Model choices were made using two forward folds entirely inside training.

covariance: conditional label-permutation p=0.835, Holm p=1.000; predefined signal gate **failed**.

eegnet_style: conditional label-permutation p=0.608, Holm p=1.000; predefined signal gate **failed**.

Conditional null shuffles test labels within their original balanced blocks. Exchangeability is assumed, not proof of randomized acquisition. Nine full training-label permutation refits with repeated inner selection were limited prospectively to participant01 thinking; these are coarse implementation controls.

Stimuli and speaking comparisons use participant01 only. Listening/spoken success may reflect sensory input or muscles. Rest follows speech. Nuisance models use raw amplitudes/means/line fractions/flags; failure of these controls cannot certify neural specificity.

Independent verification recovered 352 scores from 16814 real-trial predictions, confirmed source labels, disjoint real trials and inner folds, and processed-cache/source-snapshot hashes.

All full predictions and EEG caches are on D:. Initial_v1 is preserved. No splicing comparison should proceed without the signal gate and specificity review. The separate Kara format pilot has no classifier results.

## Prespecified sensitivities and scheduling audit

- covariance, strict, seed17: 6.00% [4.46, 7.69]%; 21 people.
- eegnet_style, primary, seed17: 6.05% [4.56, 7.54]%; 21 people.
- eegnet_style, primary, seed29: 5.06% [3.77, 6.45]%; 21 people.
- eegnet_style, primary, seed43: 6.05% [4.86, 7.34]%; 21 people.
- eegnet_style, strict, seed17: 6.15% [4.91, 7.39]%; 21 people.

All21 participants share the same trial-ID-to-prompt mapping wherever their recordings overlap. A supplementary training-only modulo16 scheduling lookup reaches10.42% in the20 full recordings and12.50% in the shorter participant12 recording, without EEG. Modulo32 yields4.17% and14.58%, respectively. These periods were examined after the baseline experiment and are not tuned primary models. The common schedule prevents interpreting repeated order-control scores across people as independent evidence; it further limits imagery-specific inference from a small above-chance score.

The rest-phase CNN mean7.74% [6.55,8.83]% is exploratory and rest follows spoken speech. Thinking does not outperform it. One-person listening/spoken models remain near chance. None of these observations demonstrates that all imagined-speech EEG lacks information.
