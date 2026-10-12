# Reproduction results (fixture lexicon, R=2000)

## Microgap vs. no-microgap: CMRO2 (montage-mean effect, Monte Carlo over microgap draws)
| sentence | no-mg | mg mean | effect | 95% CI | P(effect>0) | p_MC |
|---|---|---|---|---|---|---|
| trial1 | 2.1234 | 2.1328 | +0.0095 | [+0.0054, +0.0136] | 1.000 | 0.0010 |
| trial2 | 2.1249 | 2.1280 | +0.0031 | [+0.0009, +0.0053] | 0.997 | 0.0070 |
| trial3 | 2.1325 | 2.1291 | -0.0034 | [-0.0075, +0.0012] | 0.074 | 0.1499 |

## Consistency: synthetic composite vs. real-vs-real leave-one-out baseline
| metric | real-vs-real median [range] |
|---|---|
| logpsd_r | 0.987 [0.941, 0.996] |
| logpsd_rmse_db | 3.019 [0.643, 9.122] |
| relband_mad | 0.024 [0.010, 0.136] |
| src_raw_rho | 0.942 [0.350, 0.986] |
| src_dc_rho | 0.868 [0.258, 0.939] |
| cmro2_profile_rho | 0.946 [0.835, 0.988] |

### trial1
* no_microgap: logpsd_r=0.658, src_raw_rho=0.449, cmro2_profile_rho=0.988
* microgap_seed0: logpsd_r=0.672, src_raw_rho=0.139, cmro2_profile_rho=0.979
### trial2
* no_microgap: logpsd_r=0.673, src_raw_rho=-0.101, cmro2_profile_rho=0.979
* microgap_seed0: logpsd_r=0.666, src_raw_rho=-0.029, cmro2_profile_rho=0.994
### trial3
* no_microgap: logpsd_r=0.680, src_raw_rho=0.412, cmro2_profile_rho=0.982
* microgap_seed0: logpsd_r=0.673, src_raw_rho=0.082, cmro2_profile_rho=0.991

## Phoneme identification (epoch 0 vs. held-out epochs, 39-way)
* accuracy = 0.128 (chance = 0.026, real-data ceiling = 0.174, permutation p = 0.0010, n_perm=2000)