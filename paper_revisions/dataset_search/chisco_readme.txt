# Chisco Dataset

This dataset is a Chinese imagined speech dataset with five participants, identified as sub-01 to sub-05. The dataset includes raw data and preprocessed data in both fif and pkl formats. Information also can be found in https://github.com/zhangzihan-is-good/Chisco

## Supplementary Information

The initial dataset release encompassed data from three participants (sub-01 to sub-03) as detailed in related Chisco publications. Subsequently, data from two additional subjects (sub-04 and sub-05) were incorporated. During the interval between the original dataset release and the addition of the new data, the BIDS protocol underwent updates. To preserve the integrity of the data processing code presented in our publications, the supplementary data continue to adhere to the previous version of the BIDS protocol. Consequently, the BIDS validator on our website may report errors; however, these do not compromise the usability of the dataset.

Future releases will include data from sub-06 and sub-07, who participated under a new experimental paradigm. These will be published as part of a new dataset, Chisco 2.0. We invite you to stay tuned for further updates.

## Dataset Structure

### Root Directory

- `dataset_description.json`
- `participants.tsv`
- `README`
- `derivatives/`
- `sub-01/` to `sub-05/`
- `textdataset/`
- `json/`

### Raw Data

The root directory contains folders `sub-01` to `sub-05` with raw data. Each participant's folder contains 5-6 session folders, corresponding to data collected over 5-6 days.

### Preprocessed Data

Preprocessed data is stored in the `derivatives` folder in both fif and pkl formats.

### Text Data

The `textdataset` folder and `json` folder contain text data used to stimulate the participants.

### File Structure
```
/Chisco
    /sub-01
        /ses-01
            /eeg
                sub-01_ses-01_task-imagine_eeg.edf
        ...
    /sub-02
        ...
    /sub-03
        ...
    /derivatives
        /fif
            /sub-01
                ...
            /sub-02
                ...
            /sub-03
                ...
        /pkl
            /sub-01
                ...
            /sub-02
                ...
            /sub-03
                ...
    /textdataset
        ...
    /json
        ...
    dataset_description.json
    README
    participants.tsv

```
## License
This dataset is licensed under the CC0 license. You are free to use the dataset for non-commercial purposes, but the original author needs to be properly indicated.

## Citation
If you use this dataset in your research, please cite the following link:

https://github.com/zhangzihan-is-good/Chisco

## Contact Information
For any questions, please contact the dataset authors.
Thank you for using the Chisco!