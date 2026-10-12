# Research viability assessment and next evidence plan

3 October 2026. This is a research decision brief, not a claim of new experimental results. It accompanies the v1.27 author-review package without changing that frozen package.

The revisions improve correctness, traceability, and the boundaries of the claims. They do not establish that the method predicts imagined-sentence EEG, measures metabolism, improves decoding, or is more useful than simpler alternatives. My assessment is that the current contribution remains weak: inserting different recorded segments changes the composition of the signal, so changes in downstream averages are largely expected. A stronger paper needs a result that distinguishes the proposed method from plausible controls and matters for a defined use.

**What the reviewers actually object to.** Reviewer 4 emphasizes validity and interpretation, rather than explicitly rejecting the paper for low significance. Reviewer 3 asks for consistency comparisons and clarification of “fractal.” Reviewer 4 requests independent biological validation or narrower claims, appropriate statistical units, and fully regenerated results. Both reviewers accept the language. Additional prose editing is therefore unlikely to resolve the central problem. See the unchanged source comments in [the extracted review](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_work/thinkTankComments2.txt).

PLOS ONE does not base editorial decisions on perceived impact, but it requires technically rigorous work with supported conclusions. Its methods/software criteria also ask for demonstrated purpose and useful advantages over applicable alternatives. An honest prototype description alone does not establish that advantage. [Publication criteria](https://journals.plos.org/plosone/s/criteria-for-publication), [methods and software guidance](https://journals.plos.org/plosone/s/submission-guidelines#loc-methods-software-databases-and-tools).

**What changed in the current working revision.**

| Area | Completed change | Scientific limit that remains |
| --- | --- | --- |
| Study framing | Text-guided EEG composition and exploratory modeling; unsupported decoding, fractal, and biological-realism claims removed | The narrower claims still need a useful research purpose |
| Input selection | Marker-based task blocks, nominal 250 Hz, an archived stimulus crosswalk, and recorded intertrial waits | Acquisition version, physical channel order, participant linkage, reuse, and pronunciation approximations need author verification |
| Analysis | Separate within-session calibration, construction, and reference trials; duration controls and sensitivities; regenerated tables and four figure sets | These are not independent full-sentence biological references |
| Statistics | Electrode-based biological tests and treatment-draw p-values withdrawn; simulation intervals and Monte Carlo error reported | More simulations do not add participants or biological replication |
| Physiology | Dimensionless gamma index is primary; the absolute CMRO2 scale is imposed and the eight outputs share one driver | No independent calibration or metabolic measurements |
| Reproducibility | Four editable documents, frozen inputs, draw-level results, checks, and a local author-review ZIP | Author verification and an approved stable public deposit are outstanding |

The replacement analysis changes substantial protocol choices, including event selection, mapping, construction unit, gap sources, spectral windows, and statistics. It must be presented as an audited replacement analysis, not as an unchanged rerun of the original experiment. The 38 recorded numerical checks and six research tests check computations; they do not establish biological validity. Details are in [Changes and evidence](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/documents/Changes_and_evidence.md).

**Why the current evidence is insufficient.**

The 44 local files are isolated cued-stimulus recordings. All match an archived public source after line-ending normalization; initials do not prove who the participant was. The first 512 samples selected by the old counter-zero rule precede the first imagined-speech task marker in all 44 files. The archived stimulus order also differs from the application's mapping, and files 39–43 are speech stimuli rather than rest recordings. These findings affect what the historical result represents.

The replacement source comparisons use different trials from the same recording set. Broad spectral similarity to that set is a descriptive check, not proof that a constructed sentence resembles a person imagining that sentence. No matched continuous recordings of the three full sentences were identified in the supplied material.

There is a further structural limitation: averaging independently computed window features makes the current gamma/model averages invariant to reordering the selected windows. The code explicitly checks this. The main outcome therefore cannot establish sentence-order, linguistic-context, or transition-smoothing effects. Changing the block mixture can change the mean without capturing any sentence-specific physiology.

Signal quality is unresolved. The audit finds substantial power near 60 Hz in the broad gamma band and large offsets in stored units. Muscle contamination is another plausible high-frequency confound; a primary study using neuromuscular blockade illustrates that risk, but does not diagnose these recordings. [Whitham et al., 2007](https://pubmed.ncbi.nlm.nih.gov/17574912/). A 30–45 Hz sensitivity alone does not validate neural gamma.

**Recommended research question.**

Does a composition built from the correct isolated-phoneme recordings predict independently recorded EEG features of that imagined sequence better than duration-matched random, misassigned, and reordered compositions?

This question tests whether the phoneme organization adds information. A positive, replicated result could justify a limited compositional EEG model. A carefully designed negative result could identify the limits of that assumption. Neither result would automatically establish unknown-speech decoding or metabolic prediction. This is a proposed direction, not a verified novelty claim; a focused prior-art comparison must precede a new contribution claim.

**Do the work in this order.**

| Stage | Work and suggested responsibility | Decision gate |
| --- | --- | --- |
| 1. Verify the recordings | John and the acquisition/data custodian confirm protocol, participant/session identifiers, electrode order, units/gain, markers, consent/reuse, and signal-quality review | If usable task EEG cannot be established, restrict the present evidence to a software demonstration |
| 2. Test whether phoneme identity adds reproducible information | Analysis team compares held-out real phoneme trials/sessions with same-phoneme versus mismatched and pooled references; audit additional source directories before including them | Proceed with a composition hypothesis only if distinctions survive plausible cue, artifact, duration, and session confounds |
| 3. Freeze a sequence-validation protocol | Authors choose the claim, primary endpoint, strongest baseline, exclusions, timing treatment, and a meaningful effect/precision target before inspecting the final test data | There must be an evaluation that could falsify the proposed advantage |
| 4. Obtain independent sequence evidence | Acquisition team locates or collects approved recordings of isolated phonemes and short continuous imagined sequences from the same people across sessions, with repeated trials and matched waiting conditions | Without real sequences, sentence-level fidelity remains untested |
| 5. Compare and report | Analysis team evaluates frozen methods on untouched recordings; authors write the manuscript around the supported result | Do not interpret a small p-value or shared spectrum as sufficient utility |

Stage 2 is a feasibility check. Success could reflect stimulus presentation or recording conditions and would not prove imagined-phoneme neural specificity. Failure would undermine this particular implementation and dataset, not all compositional neuroscience. The archived upstream tree lists 816 text files under TestData and multiple recording directories, so additional sessions may be available. This is a file inventory, not a verified participant count or a determination that those files are usable. Additional isolated-phoneme files cannot replace continuous sequence recordings.

**Minimum design for stage 4.**

Start with short sequences, then words and sentences. Include sequences with the same phoneme inventory in different orders to separate composition from order. Record isolated phonemes and continuous sequences in comparable acquisition conditions, with repeated trials and at least a separate evaluation session. A one-person study can test a personalized model across sessions but cannot support population generalization. A broader claim needs independently sampled participants and an appropriate sample-size/precision plan.

Cue timing is observed; the timing of internally imagined phonemes is not automatically known. Use a declared pacing or task-level alignment protocol, document its limits, and avoid treating cue markers as direct evidence of internal speech timing. Any timing/alignment model must be selected without fitting to the final evaluation recordings.

Use the same total duration, gap schedule, available training recordings, and preprocessing budget across methods. The core comparisons are correct mapping, randomly assigned phoneme identities with matched counts, the same selected blocks in a different order, and a duration-matched pooled/average template. Compare no gaps, recorded waits, and other pause models separately. Real repeated sequences provide an empirical repeatability benchmark.

Choose one primary endpoint that can distinguish the claimed advantage, such as error in cue-aligned channel-by-frequency trajectories against held-out real sequences. Prespecify the frequencies, normalization, alignment, and aggregation; avoid relying only on whole-record averages or raw waveform phase. Report real-versus-real repeatability alongside model-versus-real error. Boundary behavior and spatial covariance can be secondary measures. Source-space metrics require verified geometry and a common justified operator.

All calibration, artifact thresholds, templates, scaling, alignment, and tuning belong to training data. No evaluation donor block may enter the composition library. For participant-level claims, estimate paired method differences across participants with repeated trials and sentences treated as nested observations; channels and synthetic draws are not independent biological samples. For a personalized study, state that uncertainty is conditional on that individual.

**If independent sequence data cannot be obtained.**

A software/benchmark direction is possible, but it requires a specific demonstrated use. One candidate is whether synthetic training data improves performance on held-out real imagined-phoneme recordings compared with no augmentation and ordinary matched augmentation. Synthetic-only accuracy is insufficient, and reused donor blocks must never leak across the split. This would be a different experiment and potentially a different paper from the current metabolic demonstration.

Existing tools already simulate EEG: [MNE's source-to-sensor example](https://mne.tools/stable/auto_examples/simulation/simulate_raw_data.html) and [SEREEGA](https://opus.lib.uts.edu.au/bitstream/10453/130158/1/krol2018jnm-sereega.pdf) establish that general simulation functionality is not itself a new contribution. They use different modeling approaches and are not automatically direct imagined-speech competitors. Compare the closest appropriate methods for the chosen purpose and explain any exclusions. A GUI, an LLM text input, and more visualizations do not substitute for demonstrated task value.

**Keep metabolism as an illustration for now.**

For the recommended next study, EEG fidelity should be the central endpoint. Independent physiology and a justified calibrated model would be needed to promote the metabolic outputs to biological estimates. Agreement between two outputs passed through the same transform does not provide that validation. Ordinary hemodynamic measurements should not be casually described as direct CMRO2 ground truth. Consult the model/acquisition experts before designing that separate validation arm.

**Current publication position.**

The v1.27 package is an author-review draft with a more transparent exploratory result. It is not ready to send as validated sentence biology. The team can pursue a narrower corrected revision after resolving author-verification items, but that route does not settle the utility concern. My recommendation is to prioritize stages 1 and 2, then choose either independent sequence validation or a concrete software-utility experiment. If the journal deadline conflicts with that work, the corresponding author can request an extension; no message or submission has been sent.

The practical decision is whether phoneme-specific organization adds value beyond simple controls. If it does, the project has an evidence-based direction to develop. If it does not, retain the useful software and report the limits rather than sustaining the current central biological interpretation.
