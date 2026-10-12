# Response to the second round of peer review

Working revision 1.27 — 3 October 2026 — Author verification required

**PROPOSED RESPONSE FOR AUTHOR REVIEW. This letter must be finalized after metadata and ethics confirmation and public deposition. Do not submit statements of completed release work until they are true.**

We thank the reviewers for identifying the remaining interpretive and computational problems. We have replaced the numerical analysis and substantially narrowed the manuscript’s claims. The replacement draft reports a text-guided composition experiment, distinguishes simulation variation from biological replication, and presents consistency diagnostics with their limits. A data audit also identified event-selection and file-order problems beyond those described in the earlier code corrections. All current numerical tables and figures are generated from the local round2-v1.27-working package. The manuscript contains explicit author-verification items that must be resolved before resubmission.

## Reviewer 3 Comment 1 Terminology around fractal

Comment: Make the Introduction terminology consistent and either define or justify fractal at first use or remove the quotation marks.

Response: We removed fractal and self-similar claims from the replacement manuscript. The implemented interpolation and spherical display do not establish a fractal construction or a measured self-similarity property. The Introduction now states the computational aim directly, and the Methods distinguish fixed source models from display resolution. Location: Section 1 and Sections 2.5 and 2.7.

## Reviewer 3 Comment 2 Consistency across three domains

Comment: Supplement the experiment with synthetic-versus-real consistency analyses in time-frequency, source space, and metabolic metrics.

Response: We added descriptive comparisons in all three requested domains, using marked task trials 3–5 as a within-session reference while constructing composites from trial 2. Spectral comparisons use a common 2 Hz grid and report log-PSD correlation and RMSE. The source-space sensitivity uses the same covariance, centering, referencing, and inverse operator for every comparison. Model-output comparisons report channel-profile rank correlation and absolute difference on a common calibration scale. Table 5, Figures 3–4, and consistency.csv contain the outputs. Location: Sections 2.7 and 3.4.

We explicitly limit these comparisons. The reference remains from the same recording-source set. The spherical inverse is not a validated replacement for the GUI’s fsaverage BEM and remains conditional on electrode assignment verification. The metabolic comparison uses computed proxies from EEG, not independently measured metabolism. No matched continuous recording of the three full text responses was supplied. High correlations therefore do not establish statistical indistinguishability or biological equivalence. This correction replaces the stronger wording in the draft response.

## Reviewer 4 Comment 1 Biological interpretation and reconstruction framing

Comment: Insertion-induced output changes do not validate biological realism; avoid enhanced biological variability and clarify that the workflow is a text-to-EEG/metabolic simulation.

Response: We revised the title, Abstract, Introduction, Discussion, and Conclusions to describe text-guided EEG composition and exploratory model propagation. The known text determines the segment sequence, and no unknown utterance is decoded. We removed enhanced biological variability, natural-transition smoothing, and related clinical or consciousness claims. The primary insertion pool now contains recorded instructed-wait intervals, with its timing and sampling rule exposed. These are not natural sentence pauses. We added a duration-matched repeated-task control and sensitivity analyses showing that file mapping, sample composition, and scaling affect the estimated changes. Location: Sections 1, 2.3, 2.6, 3.2–3.3, 4, and 5.

The Methods and Discussion explain that no spectral window crosses a segment join and that aggregate modeled means are invariant to reordering the selected windows. Consequently, this experiment cannot establish improved transition continuity or sentence-context physiology. Continuous-sentence EEG and independent metabolic measurements are identified as necessary future validation data rather than claimed as completed experiments.

## Reviewer 4 Comment 2 Statistical unit dependence and inconsistent tables

Comment: Channels are correlated, derived variables share one driver, and the reported t, p, effect-size and Trial 2 results require reconsideration.

Response: We withdrew the paired-channel inferential tables and all corresponding p-value claims. The primary summaries describe 2,000 seeded algorithmic realizations for each fixed text, conditional on the same construction blocks. We use mean changes, simulation SDs, empirical 2.5th–97.5th percentile intervals, and Monte Carlo standard errors. A realization is not a biological replicate, and the simulation interval is not a participant-population confidence interval. The earlier draft’s two-sided tail count over treatment draws did not sample a null distribution and therefore is not retained as an exact Monte Carlo hypothesis-test p-value. Location: Section 2.6 and Tables 3–4.

We also replaced the primary outcome with a dimensionless normalized gamma index. The imposed CMRO2 scale and all downstream transformations are labeled as illustrative model outputs. All eight variables depend on one scalar driver; several are affine transformations under fixed constants. The other variables are retained as dependent supporting outputs, without separate significance claims. Location: Sections 2.4–2.5 and 3.5.

The original zero-valued and inconsistent statistics are removed, and current results are generated directly from simulation_summary.csv and results.json. We cannot determine the exact historical cause of every old table cell from the supplied artifacts and do not attribute them to a single formatting or transcription defect. The revised Abstract, Results, tables, and response all use the same replacement analysis and do not repeat the prior Trial 2 significance assertion.

## Reviewer 4 Comment 3 Complete regeneration with corrected code

Comment: Regenerate manuscript tables, statistics, and supplementary figures using the corrected implementation and one reproducible version.

Response: We regenerated all five main numerical/input tables, the model-constant table, four replacement figures, simulation draws, and consistency outputs for the replacement draft from round2-v1.27-working. The run records 38 passing numerical checks. Distinct sample blocks are selected using paired trial markers, rather than repeated nominal trial slices or packet-counter zeros. Fixed-window spectral extraction avoids the old trial-indexing path. The quantitative source diagnostic uses direct fixed-operator maps; historical distance-interpolated cortical renderings are removed from the numerical evidence. The absolute-to-fractional CMRO2 conversion is explicit. Location: Sections 2.1–2.8 and S1 Data.

The deeper audit found that all 44 local files match the public source files after line-ending normalization, the old first-zero selections precede the first imagined-speech onset in every file, and the application’s file mapping conflicts with the public stimulus order. The replacement protocol therefore changes event selection, nominal sampling rate, pronunciation lookup, file mapping, insertion pool, and statistical interpretation. It is presented as a replacement exploratory analysis rather than an unchanged historical experiment with updated decimals.

**FINALIZATION REQUIRED: verify the applicable acquisition protocol and electrode order, approve the crosswalk and declared pronunciations, complete participant/ethics and reuse statements, and deposit the exact approved package under a stable release identifier. Insert that identifier into the manuscript and response before asserting a public reproducible release. The current local generation does not complete the public-release requirement.**

## Files and manuscript locations

The author-review bundle includes the replacement manuscript, this response, a detailed change record, an author decision brief, four figure sets, the analysis code and inputs, and the complete numerical outputs. Section and table numbers in this letter refer to the replacement manuscript. The final marked-up journal copy should compare against the exact manuscript currently on file with PLOS, not an assumed earlier version.
