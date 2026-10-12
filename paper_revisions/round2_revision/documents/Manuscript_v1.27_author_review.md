# Text guided EEG composition and exploratory neurovascular modeling using recorded imagined speech trials

Working revision 1.27 — 3 October 2026 — Author verification required

Gowrish Rajagopal¹, Aniket Chaudhari¹, John LaRocco²*, Jinyi Xue³, Eric Zachariah¹, Shwetan Varanasi¹, and Douglas Rothman⁴

¹ College of Engineering, Ohio State University, Columbus, OH, USA. ² Ronin Institute for Independent Scholarship 2.0, Sacramento, CA, USA. ³ College of Arts and Sciences, Ohio State University, Columbus, OH, USA. ⁴ School of Medicine, Yale University, New Haven, CT, USA.

*Corresponding author: john.larocco@ronininsitute.org. Author names, affiliations, and contact are retained from version 1.26 and require author confirmation.

**AUTHOR REVIEW NOTICE: This replacement draft reports a new exploratory reanalysis. Confirm the acquisition channel order, stimulus protocol and crosswalk, participant provenance, ethics and reuse terms, and public release identifier before submission. The local package has not been deposited or submitted.**

## Abstract

Text-driven assembly of recorded electroencephalographic (EEG) segments offers a controllable way to examine how composition choices propagate through a physiological simulation. We revised a framework that links a retrieval-augmented language model to pronunciation lookup, EEG assembly, spectral features, and illustrative neurovascular outputs. An audit of 44 source files recovered 220 paired event intervals, identified a packet counter previously used as an event anchor, and exposed a mismatch between the application file order and the public stimulus order. The replacement analysis used marked task intervals at a nominal 250 Hz and compared three fixed texts with and without inserted segments from recorded instructed-wait intervals. Across 2,000 seeded realizations per text, mean changes in a normalized gamma index were -0.0053, -0.0060, and -0.0035; their central 95% simulation intervals were [-0.0106, 0.0011], [-0.0102, -0.0014], and [-0.0103, 0.0048]. These intervals describe algorithmic variation rather than uncertainty about a population effect. Comparisons with separate trials from the same recording set showed similar coarse spectral and channel profiles but variable source-space agreement. Duration controls, alternate mappings, and normalization choices changed the effect estimates. The framework therefore demonstrates reproducible propagation of assembly choices through a model. Validation against continuously recorded imagined sentences, independently measured metabolism, and verified acquisition metadata remains necessary.

Keywords: imagined speech; electroencephalography; text-guided simulation; neurovascular modeling; reproducibility; sensitivity analysis

## 1 Introduction

Imagined speech research examines neural activity associated with internally produced speech. In a decoding experiment, a model uses neural measurements to infer an utterance or a linguistic property. Studies of EEG phoneme classification and noninvasive language reconstruction establish relevant measurement and evaluation problems [1,2]. The present framework instead starts with known text and uses its pronunciation to select recorded EEG segments. Its output is a constructed signal whose linguistic order is supplied by the text. No decoder accuracy or recovery of an unknown utterance is evaluated.

This distinction matters because a segment associated with an isolated speech cue need not preserve its neural response when placed next to other segments. Continuous sentence production contains contextual, timing, and coarticulatory effects that simple concatenation does not model. Inserting additional segments can change spectral summaries by changing the sample composition and duration. Such a change alone does not establish a more realistic biological signal.

A second distinction concerns physiological interpretation. The software maps a normalized EEG gamma feature onto a selected oxygen-consumption scale and then applies algebraic relations inspired by a neurovascular homeostasis model [3]. Those relations allow the propagation of an assumed driver through a consistent set of outputs. They do not calibrate scalp gamma power to oxygen consumption or measure blood flow, tissue pH, gas tensions, or lactate. All resulting quantities remain conditional on the imposed mapping and constants.

The purpose of this study is to characterize the computational behavior and limitations of this framework. We ask how inserting recorded waiting-period segments changes a primary normalized gamma index, whether the change depends on assembly and scaling choices, and how constructed signals compare with separate source trials in spectral, source-space, and model-derived channel profiles. A provenance audit and a replacement numerical analysis address defects identified during peer review. We use descriptive simulation summaries and make the distinction between algorithmic replication and biological replication explicit.

## 2 Materials and methods

### 2.1 Source recordings and provenance

The local recording set comprises 44 files named DLR_0_1.txt through DLR_43_1.txt. Every file contains 32 columns. After converting CRLF line endings to LF for comparison, the Git blob hash of each local file matches a file in the public ghostTalkerAlpha repository at commit 40fe0cf64e79916ced50cb82d7d0ee03f77b41c3, including the directory TestData/DLR_Tests/3-21-2023 [4]. This establishes file provenance and does not independently establish participant identity. The previous manuscript describes a single-participant subset; the exact participant linkage and reuse authorization require confirmation from the data custodian.

Column 0 takes values from 0 to 255 and repeatedly wraps to zero. It is a packet counter, not an event-time coordinate. Column 30 contains timestamps, and column 31 contains two occurrences each of markers 1 through 5. The archived acquisition script inserts the same marker at the beginning and end of each instructed two-second imagined-speech interval. We therefore record each start and end row explicitly and extract the first 250 samples following each start for the present analysis. These one-second blocks are analysis selections within marked trials, not additional independent trials.

The original publication and the archived Wi-Fi acquisition script specify a nominal sampling rate of 250 Hz [1,4]. Across the 220 marker pairs in the local files, the median ratio of sample count to timestamp duration is 249.84 samples/s. Individual ratios range from 236.55 to 256.46; timestamp delivery irregularity and possible packet loss must be distinguished before biological interpretation. The repeating 256-value counter does not establish a 256 Hz sampling rate. The replacement analysis uses the documented nominal 250 Hz rate.

The application assigns the first 39 files to an alphabetical ARPAbet inventory and uses files 39–43 as auxiliary segments. The public acquisition script and stimulus slides use a different order. For example, slide index 0 illustrates the vowel in feel/read, index 28 illustrates B, and indices 39–43 illustrate NG, R, L, W, and Y. The replacement crosswalk follows the archived slides. AH0 is mapped to the again/under stimulus and stressed AH to nut/cut. AA and AO use the dark/are and roar/morning stimuli. These mappings involve dialect approximations and are not evidence of a validated phoneme decoder. The applicable acquisition version and physical electrode-to-column order remain author verification items.

### 2.2 Language model and frozen text inputs

The desktop application obtains responses from a retrieval-augmented conversational model using an interview/autobiography corpus. The corpus is split into 500-character chunks with 50-character overlap, embedded using text-embedding-3-large, and retrieved through a FAISS index. The configured response model is gpt-4-0125-preview. Retrieved text and conversation context guide the response; the software does not fine-tune the model weights on the autobiography. The model configuration describes the application, while the numerical experiment uses the three existing responses in Table 1 and makes no new model calls.

The full texts were frozen from the supplied version 1.26 manuscript. Their lengths exceed the response prompt’s stated 20-word target, so we do not claim that this limit was enforced in these examples. The texts are illustrative software inputs. Their content is not a measured autobiographical response or a clinical or cognitive outcome.

### Table 1 Frozen text inputs

| Input | Prompt | Recorded response |
| --- | --- | --- |
| S1 | What aspect of brain-computer interfaces (BCIs) do you find most compelling or exciting? | The potential for BCIs to restore lost functions, like movement or speech, in individuals with disabilities. It’s a tangible bridge between technology and humanity’s resilience. |
| S2 | Describe your favorite type of cuisine, including specific dishes, cultural significance, and any memorable experiences you’ve had with it. How does it reflect your personal tastes or background? | Vietnamese cuisine resonates deeply with me — its balance of flavors, emphasis on fresh ingredients. Pho and banh mi stand out, embodying simplicity and depth. A meal in Hanoi, streetside, was transformative. It mirrors my preference for complexity within simplicity, a reflection of my own journey through diverse cultures and the pursuit of nuanced understanding. |
| S3 | Hello, how are you? | Navigating through a maze of new protocols and faces, all while grappling with the humidity outside. It was invigorating yet daunting. |

### 2.3 Pronunciation lookup and assembly

Pronunciations were taken from CMUdict at commit 74790861f652b15e4ac49015a90074ad62a27690 [5]. The package stores the exact entries used, the dictionary hash, and its license. The first dictionary pronunciation was selected. BCIs was explicitly expanded as the letters B C I followed by plural Z; streetside was expanded by concatenating the entries for street and side. These two declared editorial pronunciations require author confirmation. Missing entries cause an error rather than silently removing words. No probabilistic pronunciation fixture or unrecorded grapheme-to-phoneme fallback is used in the replacement results.

A common word tokenizer preserves internal apostrophes and supplies the same word and phoneme stream to both conditions. The primary assembly places one 250-sample marked task block per phoneme in text order. Trial 2 supplies every task block used in a composite. Repeated occurrences of a phoneme reuse that block; they therefore remain repeated samples from one recorded trial. Stress annotations inform the explicit AH crosswalk but do not create additional EEG classes.

In the insertion condition, each boundary between words receives either one or two one-second intertrial blocks with equal probability. Each block is sampled uniformly from the 44 available files. The block begins 50 samples after the end marker of trial 2 and ends 300 samples after that marker; every selected block is verified to precede the next task marker. The protocol instructs waiting between trials, but no separate marker identifies the exact onset of the wait screen. The selected blocks may therefore contain cue-switching or neural carryover effects and should not be treated as a verified resting baseline. A two-second insertion comprises two sampled one-second blocks. These are intertrial selections from an instructed-wait protocol, not naturally timed pauses from continuous sentences. Punctuation does not create extra gaps. Each realization uses its integer seed with NumPy’s default generator; seeds 0–1999 form the primary simulation ensemble.

Calibration uses trial 1 task and wait blocks. Construction uses trial 2 blocks. Reference comparisons use task blocks from trials 3–5. These roles share a recording session and a participant source; separating them prevents sample reuse between construction and reference but does not create an independent biological cohort. Row-level event tables and seed-0 assembly manifests are supplied with the numerical outputs.

### 2.4 Spectral feature and primary normalized index

Each one-second block is divided into two nonoverlapping 125-sample windows. SciPy computes a periodic-Hann, constant-detrended power spectral density with nperseg = 125, noverlap = 0, and density scaling. This is a single tapered periodogram per window, rather than an average over overlapping subwindows. Frequency spacing is 2 Hz. Gamma power is the sum of PSD values for 30 ≤ f < 100 Hz multiplied by the 2 Hz bin width. No window crosses an assembly boundary. Short and final blocks are not discarded by a four-second grouping rule.

The primary dimensionless index is u = clip[(G − a)/(b − a), 0, 1], where G is gamma power and a and b are the first and 99th percentiles of the trial-1 calibration windows pooled across task, wait, files, and channels. The recorded bounds are a = 1377.9 and b = 7.541e+07 in squared stored-signal units. Each composite is summarized by the mean of u over windows and 16 channels. Channels are repeated spatial measurements rather than independent biological replicates. Clipping and raw-signal quality diagnostics are reported in the supplementary numerical package.

The primary analysis applies no artifact rejection or line-noise removal; this choice preserves the auditable stored signals but limits physiological interpretation. Window detrending removes a constant within each window, not muscle activity, movement artifacts, drift within a window, or line noise. A 30–45 Hz sensitivity analysis avoids the 60 Hz neighborhood while retaining the same construction design and recalculating its calibration bounds. This sensitivity cannot establish artifact-free neural gamma.

### 2.5 Illustrative physiological transform

For compatibility with the model, the software sets a CMRO2 proxy C = 2.1 + u on an imposed 2.1–3.1 μmol/(g·min) scale. The model driver is the fractional change d = C/2.1 − 1, correcting the earlier use of absolute C as a fractional change. With coupling constant n = 2.28 and baseline oxygen extraction OEF0 = 0.42, OEF = OEF0(1 + d)/(1 + nd) and CBF = C/(6.9·OEF). The latter is in mL/(g·min) under the stated concentration units; multiply by 100 to express mL/(100 g·min). These equations preserve an assumed flow–metabolism relation, not a measured EEG-to-metabolism calibration.

The supplied implementation then calculates venous pH, venous pCO2, a Hill-equation capillary oxygen proxy, and CO2- and lactate-associated proton terms using its stored constants. Supplementary Table S1 gives those constants and the executable model records every expression. The implementation is an illustrative adaptation motivated by the homeostasis model of DiNuzzo and colleagues [3]; equation and parameter provenance has not been independently established for every constant. The code field DeltaHCO2 is a modeled flux term and is nonzero at the selected baseline, so it should not be read as an independently measured change from baseline.

All eight outputs are deterministic functions of the same scalar d. CBF, CMRO2, and the two flux terms are affine functions of that driver for fixed constants. Their aligned changes therefore do not constitute independent confirmations of an effect. We report u as the primary outcome and retain the transformed values as model outputs in the supporting data. No tissue metabolism measurement was available.

### 2.6 Simulation summaries and controls

For each text, the no-insertion composition is fixed. Each inserted realization is compared with that fixed baseline. We report the ensemble mean difference, standard deviation across realizations, empirical 2.5th and 97.5th percentiles, and Monte Carlo standard error of the ensemble mean, SD/√R. The percentile interval is a central simulation interval for the specified algorithm. It is not a confidence interval for a participant-population effect. The draws vary insertion choices conditional on fixed source recordings; they are not biological replicates. No paired electrode t-test, Wilcoxon test, Cohen’s d across electrodes, or hypothesis-testing p-value is reported.

A duration-matched control uses the same insertion counts but fills each added second with a repetition of the preceding word’s final task block. This distinguishes the recorded-wait pool from simply extending duration with selected task data, although both conditions remain synthetic. Sensitivities use 500 realizations per text and separately change normalization to the composite’s own minimum and maximum, the assembly unit to the final phoneme of each word, insertion sources to task files 39–43, the file mapping to the previous application order, or the gamma band to 30–45 Hz. These alternate choices are diagnostics rather than interchangeable estimates of one biological quantity.

A window-order control reverses the selected spectral-window sequence while retaining its multiset and normalization bounds. Because the aggregate outputs have no dependence on temporal order and no analysis window crosses a join, their means should remain unchanged within numerical precision. The test identifies what this analysis can measure: sample composition and scale, rather than sentence context, transition smoothness, or a physiological response to pause timing.

### 2.7 Consistency analyses

The held-out within-session reference comprises 132 one-second task blocks from trials 3–5 in the 44 files. The time-frequency analysis compares the mean windowed spectra using correlation of log PSD and root-mean-square log-power difference in dB over 2–100 Hz. Figure 4 also displays short-time spectra of an original continuous source-file fragment and a composite using the same analysis parameters. Their contents and task timing are unmatched; the plot illustrates signal structure and does not test reconstruction accuracy.

The source-space diagnostic uses a homogeneous sphere with 202 radial shell sources and fixed electrode coordinates inherited from the supplied analysis code. All compared inputs undergo mean removal within each 250-sample block and average referencing. One empirical covariance estimated from trial-1 task and wait blocks defines a common depth-weighted, noise-normalized minimum-norm operator with depth parameter 0.8 and regularization 1/9. Mean squared transformed amplitude defines a source power map; its Spearman correlation with the reference map is reported. This simplified inverse is a sensitivity model. It is not the application’s fsaverage cortical BEM, is not anatomically validated, and remains conditional on verification of electrode-to-column assignments. Source maps should not be interpreted as localizing language function.

The model-output consistency measure compares 16-channel mean normalized-gamma profiles on the common calibration scale, using Spearman correlation and mean absolute index difference. The corresponding CMRO2 proxy is an affine rescaling and has the same rank correlation. Calling this a comparison with real metabolism would be incorrect: both profiles are computed from EEG by the same imposed model.

For descriptive context, the mean reference profile for each file is compared with the mean of the other 43 files, using the held-out task blocks. The resulting range is not a null distribution or an equivalence margin. A high correlation, including one above the observed reference range, does not demonstrate equivalence or statistical indistinguishability. Seed 0 provides the displayed composite comparisons; all Monte Carlo draws are retained for the primary insertion analysis.

### 2.8 Software verification and version control

The replacement analysis package is designated round2-v1.27-working. It records input hashes, event rows, pronunciation entries, assembly plans, every simulation draw, summary tables, and figure inputs. The current run passed 38 recorded numerical checks, including cached-versus-materialized features and aggregate invariance to window order. These checks establish specific computational properties, not biological validity. This batch reanalysis replaces all numerical tables in the prior draft; it is not a claim that the desktop GUI or historical MNE animations have been regenerated or validated. No historical animation is used as quantitative evidence in this revision.

### 2.9 AI assistance in revision and analysis development

OpenAI Codex assisted with manuscript and code auditing, source-metadata retrieval, development of the replacement analysis and plotting scripts, and drafting of revised prose and the reviewer response. The scripts were executed on the supplied recordings; plots were rendered from recorded inputs and calculated outputs using standard plotting software. No generative image model created the research figures. [AUTHOR COMPLETE: describe how the authors independently checked the code, numerical results, sources, figures, and interpretations and approved the revised content. Automated checks in the working package do not substitute for author verification.] This disclosure is included for review under the PLOS AI-use policy: https://journals.plos.org/plosone/s/ethical-publishing-practice#loc-artificial-intelligence-tools-and-technologies.

## 3 Results

### 3.1 Provenance and event audit

All 44 files contained five paired event intervals. In every file, the first 512 samples selected after the first packet-counter zero ended before the first task start marker. The earlier construction anchor therefore did not select a marked imagined-speech interval. The file-order discrepancy and the selection of speech stimuli as auxiliary gaps provided additional reasons to replace the previous numerical experiment. Table 2 records the auditable changes. They also prevent assigning the old tables’ discrepancies to a single transcription or formatting cause.

### Table 2 Data and implementation audit

| Item | Observed evidence | Replacement treatment |
| --- | --- | --- |
| Event selection | 44/44 old 512-sample selections precede first task marker | Select first 250 samples after each marked task onset |
| Sampling | Published/script nominal 250 Hz; packet counter wraps at 256 | Use 250 Hz and retain timestamp diagnostics |
| File order | B is acquisition index 28; application index 6 | Archived-slide crosswalk with explicit vowel approximations |
| Auxiliary files | Acquisition indices 39–43 are NG R L W Y stimuli | Primary insertions use recorded instructed waits |
| Pronunciation | Missing fixture in supplied v1.26 package | Frozen CMUdict entries and two declared overrides |
| Replication | One recording-source set and correlated channels | Describe conditional algorithmic variation without biological tests |

### 3.2 Changes produced by recorded-wait insertion

Table 3 summarizes the primary dimensionless index differences. The mean changes were negative for all three inputs. The interval for S2 was entirely negative under this particular simulation, while those for S1 and S3 crossed zero. This describes the range of insertion outcomes under the fixed recording set and construction rule; it does not establish a statistically significant biological reduction in oxygen consumption.

### Table 3 Primary normalized gamma simulation results

| Input | No insertion mean | Mean change | Central 95% simulation interval | MCSE of mean |
| --- | --- | --- | --- | --- |
| S1 | 0.10193 | -0.00529 | -0.01062 to 0.00110 | 0.000068 |
| S2 | 0.10251 | -0.00603 | -0.01024 to -0.00135 | 0.000051 |
| S3 | 0.09088 | -0.00354 | -0.01032 to 0.00477 | 0.000087 |

The no-insertion compositions contained 136, 259, and 92 one-second blocks. Mean inserted lengths were 171.95, 338.54, and 121.93 seconds. Mean wait-minus-duration-control differences were -0.00815, -0.01199, and -0.00149. Thus the wait pool and the repeated-task duration control produced different feature mixtures. A comparison that ignores source composition cannot attribute these differences solely to pause duration.

### 3.3 Sensitivity to assembly and normalization

### Table 4 Mean index changes under alternate choices

| Choice | S1 | S2 | S3 | Draws per input |
| --- | --- | --- | --- | --- |
| Recorded waits and fixed calibration | -0.00529 | -0.00603 | -0.00354 | 2000 |
| Own minimum and maximum | -0.00375 | -0.00436 | -0.00194 | 500 |
| Final phoneme per word | -0.02432 | -0.02954 | -0.00240 | 500 |
| Task files 39–43 as insertions | 0.00041 | 0.00024 | 0.00317 | 500 |
| Previous application mapping | 0.00131 | 0.00181 | -0.00038 | 500 |
| Gamma restricted to 30–45 Hz | -0.00538 | -0.00587 | -0.00337 | 500 |

The final-phoneme-only choice yielded substantially larger negative means for S1 and S2 than the primary all-phoneme choice. The previous file order and auxiliary-speech insertion pool yielded different magnitudes and, for some inputs, different mean signs. Restricting the feature to 30–45 Hz preserved the direction of the primary mean changes in this run. The result is therefore conditional on an explicit construction and normalization protocol, rather than an invariant property of the text or a validated physiological effect of microgaps.

The aggregate window-order controls passed for every configuration and input. Reordering the same analyzed windows preserved the modeled means. This confirms the absence of temporal-order sensitivity in the reported aggregate outcomes and limits any interpretation involving linguistic context or biological transition smoothing.

### 3.4 Consistency with separate recorded trials

### Table 5 Descriptive consistency for seed 0

| Input and condition | Log PSD r | Log PSD RMSE dB | Sphere source ρ | Index channel ρ | Index channel MAE |
| --- | --- | --- | --- | --- | --- |
| S1 no_gap | 0.976 | 2.597 | 0.059 | 0.985 | 0.0371 |
| S1 wait_inserted | 0.982 | 2.229 | 0.558 | 0.997 | 0.0295 |
| S2 no_gap | 0.978 | 2.607 | 0.295 | 0.991 | 0.0377 |
| S2 wait_inserted | 0.983 | 2.331 | 0.616 | 0.997 | 0.0325 |
| S3 no_gap | 0.980 | 2.071 | 0.584 | 0.994 | 0.0260 |
| S3 wait_inserted | 0.985 | 1.808 | 0.702 | 0.994 | 0.0214 |

Across the displayed composites, log-spectral correlations with the held-out within-session mean ranged from 0.976 to 0.985; spectral RMSE ranged from 1.808 to 2.607 dB. Source-map correlations were more variable (0.059–0.702). Channel-index correlations were high (0.985–0.997), while nonzero absolute differences remained. These measures address different aspects of the construction and do not jointly establish biological realism.

The file-excluded reference comparisons are available row by row in consistency.csv. They quantify variation within this recording set rather than a reference population of imagined sentences. No continuous EEG recording matched to the three full text responses and no independently measured metabolic endpoint were available in the supplied data.

### 3.5 Signal quality and model dependence

The fraction of gamma-band power in the 58–62 Hz bins varied substantially by channel. Median fractions over task windows ranged from 0.125 to 0.806. Large stored-signal offsets and between-channel amplitude differences also warrant a dedicated acquisition and artifact review. These observations preclude treating the present gamma index as a specific measurement of neural oxygen demand. The 30–45 Hz sensitivity reduces exposure to the line-noise neighborhood but does not resolve all signal-quality concerns.

The other seven physiological outputs changed in accordance with their deterministic transform of the same index. Their complete per-realization differences and units are supplied as supporting results. They are not additional independent outcomes or separate validations of the proposed mechanism.

## 4 Discussion

The replacement analysis establishes a narrower contribution than the earlier imagined-speech reconstruction framing: a reproducible software workflow for propagating text-guided sample selection through spectral summaries and an illustrative physiological transform. It also shows why provenance and event handling must be part of that workflow. A counter wrap was an inadequate event anchor, and an index inventory was not a verified stimulus crosswalk. These problems change the scientific interpretation even when the resulting plots appear plausible.

Recorded-wait insertion changed the primary index in a design-dependent manner. The duration-matched control and sensitivity analyses show that sample composition, repeated task blocks, and the scaling rule matter. The chosen periods were instructed waits in an isolated-phoneme protocol. They do not represent naturally timed pauses between words in the tested sentences, and the synthetic sequence does not implement coarticulation or sentence-context responses. The mean changes cannot therefore be described as enhanced biological variability, a pause-induced metabolic response, or proof of a more realistic reconstructed sentence.

The consistency results demonstrate that a constructed signal can resemble its source set in coarse spectral and channel summaries while showing weaker or variable agreement under a source transformation. Common donor characteristics, correlated channels, instrumental effects, and the shared normalization rule can all contribute to high channel-profile correlations. Correlation measures shape agreement and does not test equality of distributions or biological equivalence. The simplified sphere inverse additionally depends on montage assignments and regularization, so its maps are descriptive model outputs rather than anatomical evidence.

The physiological transform is useful for exploring the consequences of assumptions, but the gamma-to-CMRO2 mapping requires independent validation. A min–max or percentile scale imposes an absolute range; it does not estimate oxygen consumption from data. Parameters such as the coupling constant and glycolytic ratio may also affect downstream outputs. The selected constants and executable equations are exposed so that future studies can revise them, test parameter sensitivities, and compare predictions with independent physiological measurements. The present paper does not claim such calibration has been completed.

The statistical revision avoids counting channels or simulation draws as biological replicates. The simulation ensemble answers a conditional question about choices made by an algorithm on a fixed recording set. Increasing its size reduces Monte Carlo error in that conditional summary but provides no additional participants. An inferential claim about imagined-speech biology would require a defined biological unit, an independently justified null or effect model, and replication at the participant or trial level with appropriate dependence handling.

The main limitations are the single recording-source subset described in the prior manuscript, unconfirmed participant linkage and physical channel ordering, dialect approximations in the pronunciation crosswalk, repeated use of fixed task blocks, incomplete artifact assessment, and the lack of matched continuous-sentence EEG and metabolic measurements. A next validation study should record the same utterances and naturally occurring or experimentally controlled pauses across participants; retain synchronized stimulus, EEG, and timing metadata; establish preprocessing before examining effects; and evaluate on participant-separated data. Biological validation of the metabolic outputs requires an independent physiological modality rather than another application of the same EEG transform.

## 5 Conclusions

The audited workflow provides an exploratory text-guided EEG composition and neurovascular-modeling demonstration. Marked-trial selection, a documented stimulus crosswalk, frozen pronunciations, and complete simulation outputs make its computations reviewable. Recorded-wait insertion changes the modeled summaries, with magnitudes and signs dependent on assembly choices. The available data do not validate reconstruction of unknown speech, sentence-level biological realism, or quantitative EEG-based metabolism. Those questions remain targets for independent experiments after acquisition and signal-quality verification.

## Declarations

### Ethics and participant provenance

This reanalysis uses existing human EEG files and involved no new participant recruitment or acquisition. The source study reports ethics approval IRB 2023H0194 and participant consent [1]. [AUTHOR CONFIRM: identify the data custodian and participant subset, confirm that these files fall under the stated approval and consent, describe any conditions on secondary reuse, and state the applicable institutional determination for this work. Do not infer an exemption from the absence of new recruitment.]

### Data and code availability

The working analysis package contains the exact pronunciation entries, event selections, input hashes, simulation draws, summary data, and figure inputs for this draft. Source EEG files are traceable to ghostTalkerAlpha commit 40fe0cf64e79916ced50cb82d7d0ee03f77b41c3 [4]. [AUTHOR ACTION: deposit the approved package and permitted source data or stable source links, assign a fixed release or archival DOI, verify reuse terms, and insert that identifier here. The live main branch alone is not the identifier for the numerical results in this draft.]

### Acknowledgments

The authors thank Prof. David Tomasko and Ron Vlcek from Ohio State University, as stated in version 1.26.

### Competing interests funding and contributions

The version 1.26 manuscript states that the authors have no conflicts of interest. [AUTHOR CONFIRM: approve the final competing-interests declaration, provide the funding declaration and funders’ roles, and verify individual contributor roles. No new funding or contribution facts have been inferred.]

## References

1. LaRocco J, Tahmina Q, Lecian S, Moore J, Helbig C, Gupta S. Evaluation of an English language phoneme-based imagined speech brain computer interface with low-cost electroencephalography. Front Neuroinform. 2023;17:1306277. https://doi.org/10.3389/fninf.2023.1306277

2. Tang J, Chen J, Xu X, Liu A, Chen X. Imagined speech reconstruction from neural signals—An overview of sources and methods. IEEE Trans Instrum Meas. 2024;73:4011721, 1–21. https://doi.org/10.1109/TIM.2024.3472830

3. DiNuzzo M, Dienel GA, Behar KL, Petroff OA, Benveniste H, Hyder F, et al. Neurovascular coupling is optimized to compensate for the increase in proton production from nonoxidative glycolysis and glycogenolysis during brain activation and maintain homeostasis of pH, pCO2, and pO2. J Neurochem. 2024;168:632–662. https://doi.org/10.1111/jnc.15839

4. ghostTalkerAlpha contributors. Source data, acquisition script, and stimulus presentation. Commit 40fe0cf64e79916ced50cb82d7d0ee03f77b41c3. https://github.com/javeharron/ghostTalkerAlpha/tree/40fe0cf64e79916ced50cb82d7d0ee03f77b41c3

5. CMUSphinx contributors. CMU Pronouncing Dictionary. Commit 74790861f652b15e4ac49015a90074ad62a27690. https://github.com/cmusphinx/cmudict/tree/74790861f652b15e4ac49015a90074ad62a27690

## Supporting information

S1 Data. Event rows, file hashes, signal-quality diagnostics, exact pronunciations and overrides, stimulus labels, seed-0 assembly manifests, all simulation draws, model summaries, consistency rows, and figure arrays in the accompanying analysis package. These files supply the points underlying every numerical table and plot in this draft.

### Supplementary Table S1 Implemented model constants

| Constant | Value | Role or qualification |
| --- | --- | --- |
| n | 2.28 | Coupling ratio |
| oef0 | 0.42 | Baseline extraction |
| pH_A | 7.4 | Implemented assumption; see executable expression |
| pO2_A | 100.0 | Defined but unused |
| pCO2_A | 40.0 | Defined but unused |
| P50 | 27.0 | Implemented assumption; see executable expression |
| h | 2.8 | Implemented assumption; see executable expression |
| BC | 26.0 | Implemented assumption; see executable expression |
| HC | 0.33 | Implemented assumption; see executable expression |
| CF | 0.14 | Implemented assumption; see executable expression |
| Rglyc | 3.1 | Assumed model value; parameter provenance requires review |
| cmro2_baseline | 2.1 | Imposed μmol/(g·min) baseline |
| O2_A_conc | 6.9 | μmol/mL, equivalently mmol/L |
| CO2_A_conc | 25.6 | mmol/L |
| fCO2_plasma | 0.66 | Implemented assumption; see executable expression |
| fVplasma | 0.55 | Implemented assumption; see executable expression |
| pka | 6.1 | Implemented assumption; see executable expression |
| alpha_CO2 | 0.0307 | Implemented assumption; see executable expression |

## Figure legends and replacement figures

![Figure 1. Replacement workflow and within-session data roles. Trial 1 supplies calibration and the common inverse covariance; trial 2 supplies task and wait construction blocks; trials 3–5 supply reference task blocks. These partitions do not constitute independent participants. Plotting scripts were developed with OpenAI Codex assistance; see Section 2.9.](../figures/Fig1_workflow.png)

Figure 1. Replacement workflow and within-session data roles. Trial 1 supplies calibration and the common inverse covariance; trial 2 supplies task and wait construction blocks; trials 3–5 supply reference task blocks. These partitions do not constitute independent participants. Plotting scripts were developed with OpenAI Codex assistance; see Section 2.9.

![Figure 2. A, Mean index changes and empirical central 95% simulation intervals for 2,000 primary realizations. B, Means under primary and alternate analysis choices; sensitivities use 500 realizations. Intervals and means describe algorithmic variation conditional on fixed recordings, not population inference. Plotting scripts were developed with OpenAI Codex assistance; see Section 2.9.](../figures/Fig2_simulation.png)

Figure 2. A, Mean index changes and empirical central 95% simulation intervals for 2,000 primary realizations. B, Means under primary and alternate analysis choices; sensitivities use 500 realizations. Intervals and means describe algorithmic variation conditional on fixed recordings, not population inference. Plotting scripts were developed with OpenAI Codex assistance; see Section 2.9.

![Figure 3. A, Mean spectra for the held-out reference and S1 seed-0 composites. B and C, Relative modeled source powers on a simplified sphere with a common inverse operator. Source power is divided by its map mean for display. Coordinates describe a sphere, not localized cortical anatomy. Plotting scripts were developed with OpenAI Codex assistance; see Section 2.9.](../figures/Fig3_consistency.png)

Figure 3. A, Mean spectra for the held-out reference and S1 seed-0 composites. B and C, Relative modeled source powers on a simplified sphere with a common inverse operator. Source power is divided by its map mean for display. Coordinates describe a sphere, not localized cortical anatomy. Plotting scripts were developed with OpenAI Codex assistance; see Section 2.9.

![Figure 4. Short-time spectra of an original continuous source-file fragment and the first 20 seconds of a seed-0 composite. PSDs are averaged across stored channels and displayed with common limits. The original fragment contains task and wait periods and is not a matched recording of S1; this display does not measure sentence reconstruction accuracy. Plotting scripts were developed with OpenAI Codex assistance; see Section 2.9.](../figures/Fig4_time_frequency.png)

Figure 4. Short-time spectra of an original continuous source-file fragment and the first 20 seconds of a seed-0 composite. PSDs are averaged across stored channels and displayed with common limits. The original fragment contains task and wait periods and is not a matched recording of S1; this display does not measure sentence reconstruction accuracy. Plotting scripts were developed with OpenAI Codex assistance; see Section 2.9.
