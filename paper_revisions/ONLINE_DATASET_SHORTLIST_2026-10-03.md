# Online data that could strengthen the imagined-speech paper

Checked on 3 October 2026, America/New_York. Public repository metadata, primary papers, selected event/header/marker files and one archive endpoint were checked live. **Full EEG recordings have not been downloaded or analysed.** This document proposes new experiments; it does not add results to the v1.27 author-review manuscript.

**Recommendation: begin with FEIS for an English phoneme feasibility experiment. Use Kara One for a subsequent English word experiment. Chisco is the strongest verified public scalp-EEG sentence candidate, but it requires a Mandarin-specific study design and does not supply a matching isolated-phoneme bank.**

The scientific question should be whether the proposed construction adds useful information beyond matched controls on independently recorded data. Obtaining a larger dataset alone does not answer that question.

## What is available and what it can establish

| Priority and dataset | Verified recording content | Potential use for this project | Main limitation |
| --- | --- | --- | --- |
| 1. [FEIS v1.1](https://zenodo.org/records/3554128) | 21 participants with 16 English phonemes; 14-channel scalp EEG; listening, imagined and spoken phases, with rest | Test whether phoneme-specific EEG features replicate beyond the local donor recordings; evaluate an adapted augmentation method on held-out real phoneme trials | No recorded full sentences; incomplete English phoneme inventory |
| 2. [Kara One](https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html) | 14 released participant archives; seven syllabic/phonemic prompts and four English words; 64-channel scalp EEG | English word benchmark and limited smaller-unit-to-word feasibility audit | No sentences; the smaller-unit inventory does not automatically cover the four words |
| 3. [Chisco v1.1.2](https://openneuro.org/datasets/ds005170/versions/1.1.2) | Imagined Mandarin sentences; original paper: three people and 20,043 trials; current release: five people | Independent sentence-EEG study, using trial-level labels and held-out days/sentences | No matched isolated-phoneme bank or observed imagined-phoneme onsets; English ARPAbet mapping cannot be reused |
| 4. [Thinking Out Loud v2.1.2](https://openneuro.org/datasets/ds003626/versions/2.1.2) | Ten participants; four Spanish direction words; inner speech, pronounced speech and visualized movement; EEG plus external channels | Word-level replication with eye/mouth artifact checks and contrasting tasks | No sentences or isolated phonemes; three sessions are within one recording day |
| Supplemental: [VocalMind](https://zenodo.org/records/14696348) | Mandarin words and sentences in one person, with vocalized, mimed and imagined modes | Explore word-to-sentence composition if the recorded word inventory covers suitable sentence subsets | Implanted stereotactic EEG in an epilepsy patient, not scalp EEG; vocabulary overlap unverified |
| Conditional: [Synchronous EEG–fMRI v1.0.1](https://openneuro.org/datasets/ds006033/versions/1.0.1) | Three participants imagining eight English words; simultaneous EEG and fMRI | Exploratory EEG-to-BOLD prediction after a timing and artifact audit | Sampled event metadata has unexplained very short entries; BOLD is not a direct CMRO2 measurement |

The priorities reflect suitability for a first experiment, not a ranking of dataset quality. Protocol and access sources are detailed below.

## FEIS: the first experiment to run

The [author repository](https://github.com/scottwellington/FEIS/blob/main/readme.md) describes English recordings from native or advanced English speakers, three sessions for most participants and only two for participant 12. Session independence across days is not established. The fixation phase is named `articulators` in the files; it should not be mistaken for an articulation task. The continuous recording and separate phase recordings are CSV data packaged inside ZIPs.

[Zenodo v1.1](https://zenodo.org/records/3554128) exposes an open archive of 1,647,825,020 bytes, approximately 1.65 GB, and lists ODC-BY. The author documentation also describes nonprofit academic/research use. Cite the dataset and retain the supplied attribution and terms. The archive endpoint and license metadata were verified; an EEG CSV header was not inspected, so sampling rate, units and trial columns still require inspection.

**Small pilot:** participant `experiments/01/thinking.zip`, `resting.zip` and `full_eeg.zip`, plus the experiment scripts and metadata. These paths were verified in the [v1.1 repository](https://github.com/scottwellington/FEIS/tree/v1.1). Start with one participant to establish parsing and timing; use the full English cohort for scientific conclusions.

**Decision:** first test repeatable phoneme information after artifact review. If that fails, there is little basis for claiming that assembling these phonemes preserves speech-specific content. A successful phoneme benchmark would still not validate sentence reconstruction.

## Kara One: English words and a limited composition check

The [official release](https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html) has prompts `iy`, `uw`, `piy`, `tiy`, `diy`, `m`, `n`, and words `pat`, `pot`, `knew`, `gnaw`. It supplies continuous Neuroscan CNT recordings, `epoch_inds.mat`, and experimental material. Acquisition was at 1,000 Hz. Imagined trials have a five-second task interval. The release describes free academic/nonprofit use with the requested publication citation.

**Small pilot:** the `MM05.tar.bz2` download returned HTTP 200 to a HEAD request and reported 2,185,187,287 bytes, approximately 2.19 GB. No archive body was downloaded. The full collection is listed as about 24 GB on the release page.

**Decision:** inspect trial labels, session organization, reference, units, and raw signal quality before analysis. The published preprocessing used 1–50 Hz filtering; those derivatives cannot supply a broad 30–100 Hz gamma estimate. Check pronunciation and smaller-unit coverage before claiming a composed word has a matching recorded reference. Do not equate syllable prompts with a complete phoneme library.

## Chisco: real imagined sentences

The [2024 primary paper](https://www.nature.com/articles/s41597-024-04114-1) describes 6,681 Mandarin sentences per person for its original three participants, spanning 39 semantic categories. Each trial has five seconds of reading, 3.3 seconds of imagination and a rest interval. Hardware used a 128-channel cap; the recording configuration includes 125 EEG channels, external channels and a marker channel. Raw data were acquired at 1,000 Hz.

The [current release documentation](https://github.com/OpenNeuroDatasets/ds005170/blob/1.1.2/README) distinguishes the original three participants from two added later and describes five to six recording days per participant. Do not assign the original paper's trial total to all five people without counting their events. The metadata specifies CC0, while the README also mentions noncommercial use and attribution; record this wording discrepancy rather than silently interpreting it away.

**Small pilot:** `sub-01/ses-01/eeg/sub-01_ses-01_task-imagine_run-01_eeg.edf`, its event/label material, and `json/textmaps.json`. A preprocessed FIF file is also present at `derivatives/preprocessed_fif/sub-01/eeg/sub-01_task-imagine_run-01_eeg.fif`. The complete release is large, approximately 262 GB according to [NeMAR's dataset listing](https://nemar.org/dataset/on005170); download selected runs first. The [author code](https://github.com/zhangzihan-is-good/Chisco) explains the protocol and processing.

**Decision:** separate reading from imagination, validate the event-to-text mapping, and hold out sessions and sentence content according to the stated generalization question. No observed timing of individual imagined phonemes is supplied. Dividing a sentence epoch into equal phoneme slices would introduce an assumption, not recover measured phoneme events. This dataset cannot directly validate John's isolated English phoneme bank.

## Thinking Out Loud: words with artifact controls

The [primary paper](https://www.nature.com/articles/s41597-022-01147-2) reports 128 EEG and eight external channels at 1,024 Hz. The four Spanish classes are up, down, right and left. There are 2,236 inner-speech trials, alongside pronounced and visualized trials; the total across conditions must not be reported as an inner-speech-only sample. Its three sessions were consecutive within one day. External mouth/eye recordings and contamination reports are especially useful for reviewing a gamma-based method.

[Version 2.1.2](https://openneuro.org/datasets/ds003626/versions/2.1.2) lists CC0 for the dataset. The [author code](https://github.com/N-Nieto/Inner_Speech_Dataset) has a separate software license. Raw BDF files and processed FIF epochs/events are available.

**Small pilot:** `sub-01/ses-01/eeg/sub-01_ses-01_task-innerspeech_eeg.bdf`, or the corresponding files under `derivatives/sub-01/ses-01/`. Inspect the actual sampling rate in file metadata; the paper contains inconsistent descriptions of the processed rate. Filter by condition and retain trial flags.

**Decision:** use as a separate word benchmark and an artifact sensitivity analysis. The inner-voice instruction differs from articulatory speech imagery; retain that protocol distinction when interpreting replication.

## VocalMind: words and sentences, with a different recording modality

The [primary paper](https://www.nature.com/articles/s41597-025-04741-2) describes one Mandarin-speaking epilepsy patient with 140 implanted contacts, of which 110 were retained. It includes 20 words and 100 sentences in vocalized, mimed and imagined modes. Word trials were repeated approximately six times and sentence trials twice per mode. The raw recording rate is 1,000 Hz; processed features include normalized high-gamma amplitude and must not be treated as raw scalp EEG power.

The [Zenodo release](https://zenodo.org/records/14696348) is open and lists CC BY 4.0. Its raw imagined-word ZIP is 188,671,934 bytes and raw imagined-sentence ZIP is 520,323,305 bytes, approximately 189 MB and 520 MB. Raw CSV and processed NPY releases are separate. Download those two archives and labels only if pursuing this supplemental experiment; the complete bundle is approximately 3.7 GB.

**Decision:** audit how many target sentences can actually be assembled from the recorded word inventory. The same person recorded both levels, which is useful, but complete vocabulary coverage is unverified. Results would be a single-patient intracranial feasibility study and cannot establish the validity of the scalp pipeline.

## Simultaneous EEG–fMRI: an exploratory physiology option requiring audit

The [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12681749/) describes three people, two sessions, and eight English words. EEG was acquired at 5,000 Hz during fMRI with a two-second repetition time. The public EEG has undergone MRI gradient and pulse-artifact correction; unprocessed EEG is available only by request. Participant 1's first EEG session is absent. The fMRI contains initial dummy volumes excluded from EEG, so alignment needs explicit verification.

[OpenNeuro v1.0.1](https://openneuro.org/datasets/ds006033/versions/1.0.1) lists CC0. A sampled native BrainVision header confirms prior scanner/pulse correction and a 200-microsecond sampling interval. This verification did not include the binary EEG or MRI images.

**Observed metadata issue, limited to the sampled participant/session:** in `sub-02/ses-01`, the first 140 of 319 entries in both EEG and fMRI event TSVs have durations below 0.1 seconds, including word/rest labels. Subsequent entries have two-second task and roughly twelve-second rest durations. For example, an early `father` entry has duration `0.013`. These early entries need an explanation against the protocol, native acquisition markers and task logs. Do not rescale the whole file. The exact values, source hashes and native header/marker receipts are saved in [the timing audit](C:/Personal_Coding/thinktank-clearmind/paper_revisions/dataset_search/event_timing_audit.json). This is not a finding that every session is unusable.

**Decision:** keep this as a conditional later experiment. An EEG-to-BOLD model must beat a task-timing model and artifact/other-band controls on held-out observations. Three participants support exploratory work. BOLD is not ground-truth CMRO2, CBF or the other metabolic variables in our manuscript; this dataset cannot by itself calibrate those eight outputs.

## Other leads and access limits

| Lead | Why it matters | Verified access status |
| --- | --- | --- |
| [English continuous-speech study, Sharon et al., 2020](https://www.surlab.org/wp-content/uploads/2020/09/Sharon-et-al-IEEEAccess-2020.pdf), with [2025 follow-up](https://www.surlab.org/wp-content/uploads/2025/03/Harnessing_the_Multi-Phasal_Nature_of_Speech-EEG_for_Enhancing_Imagined_Speech_Recognition.pdf) | Nine participants; short English sentences in hearing, imagining and speaking conditions; repeated sessions/trials | Papers verified, but no public data download or reuse terms verified. Contact would be a separate action; none sent. |
| [DAIS](https://lifesciences.datastations.nl/dataset.xhtml?persistentId=doi%3A10.17026%2Fdans-xc3-66ze) | Dutch imagined vowels/word pairs; useful word benchmark | Restricted files; do not describe as immediately downloadable. |
| [T-MSPD](https://cstr.cn/31253.11.sciencedb.24416) | Mandarin EEG plus speech-muscle measurements; useful artifact-control lead | Restricted access/application; not acquired. |
| [Chisco 2.0 v1.1.1](https://openneuro.org/datasets/ds006317/versions/1.1.1) | An additional public release with two participants | Metadata/README verified, CC0 listed. Protocol-to-analysis suitability not fully verified; keep separate from Chisco v1.1.2. |

Reading or listening datasets can support other language-neuroscience questions, but they are not substitutes for imagined-speech task recordings in this validation.

## A feasible experiment that could improve the paper

These are proposed decisions, not reported results or claims that publication is assured.

1. **Establish reproducible phoneme information with FEIS.** Parse continuous data and task labels, review artifacts, use the supplied sampling rate/channel names/units, and split real trials before creating a donor bank. Fit preprocessing choices and scaling on training data only. Assess simple classification and repeatability on real held-out trials. Include lower-frequency features and controls for label-independent structure.
2. **Test a defined practical benefit.** If repeatable information exists, adapt the method to a clearly defined segment-recombination augmentation experiment. Compare training on real data alone, conventional augmentation, same-duration random donor construction, and the proposed label-guided construction. Evaluate all methods on the same untouched real test trials, with equal data/computation budgets. This is a new utility experiment, not evidence that the existing sentence-splice model reconstructs biology.
3. **Replicate an effect before enlarging the claim.** Use all suitable English FEIS participants, then a separate Kara One word benchmark. Report participant-level effects and uncertainty; synthetic draws and electrodes do not increase participant count. Hold out sessions only when their structure supports that question; hold out participants when claiming transfer to new people. Splits and tuning must reflect the claimed use.
4. **Treat sentence composition as a separate hypothesis.** Chisco supplies real imagined sentence EEG for an extension, but not the matched isolated-unit bank. VocalMind could permit a small word-to-sentence check if vocabulary coverage is sufficient, with intracranial limitations stated. For a direct English scalp test of this paper's core hypothesis, seek recordings of smaller units and independent target sequences from the same people, with reliable task timing. I did not verify a public dataset that meets all those requirements.
5. **Keep the metabolic model illustrative unless independently calibrated.** EEG–fMRI could support a separate measured-BOLD prediction study after its metadata audit. It cannot convert imposed model outputs into measured oxygen metabolism.

Choose one primary outcome before inspecting test-set results. For an augmentation paper, a suitable outcome is the participant-level change in balanced classification accuracy on real test trials, with identical train/test partitions across methods. Include label-shuffling and artifact-sensitive checks. Sequence claims need a metric that actually changes with order; our current window-averaged gamma summary cannot establish correct sentence order.

A useful negative result would also guide us: if label-guided construction performs no better than conventional augmentation or matched random construction, the paper should not present it as a validated biological reconstruction method. Dataset quantity cannot repair that claim.

## Verification receipts and acquisition boundaries

- [Public metadata checks](C:/Personal_Coding/thinktank-clearmind/paper_revisions/dataset_search/access_checks.json): URLs, HTTP results, hashes, versions, licenses and file metadata from 14 successful public API checks.
- [Sample/access checks](C:/Personal_Coding/thinktank-clearmind/paper_revisions/dataset_search/example_checks.json): two event files, an EEG sidecar, FEIS README, and a Kara One archive HEAD response.
- [Timing audit](C:/Personal_Coding/thinktank-clearmind/paper_revisions/dataset_search/event_timing_audit.json): sampled native header/marker and fMRI sidecar checks, event summaries and caveats.
- [Dataset manifest](C:/Personal_Coding/thinktank-clearmind/paper_revisions/dataset_search/shortlist.json): proposed uses, pinned release links, formats and pilot paths.

OpenNeuro GitHub trees often contain Git-annex symlinks for recordings; cloning the metadata repository alone does not retrieve the EEG. Selected files must be acquired through the dataset's download mechanism. Validate checksums, event counts and file metadata at acquisition. Current Chisco metadata was checked against the release tag; other versions are explicit above.

No author access request, collaborator message, full EEG download, manuscript alteration, submission, public deposit or GitHub write was performed for this search. The existing author verification requirements for the round-two revision remain open.
