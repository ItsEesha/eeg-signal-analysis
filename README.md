# EEG Signal Analysis for Neural Feature Extraction

An exploratory Python workflow for analysing motor-imagery EEG data from the [PhysioNet EEG Motor Movement/Imagery (EEGBCI) dataset](https://physionet.org/content/eegmmidb/1.0.0/).

This project demonstrates a research workflow for EEG preprocessing, frequency-band feature extraction, and an exploratory subject-independent machine-learning evaluation.

> **Research status:** This is an educational/research workflow. It is not a validated brain–computer interface, clinical diagnostic tool, or real-time robotic controller. No model performance claims should be made until the script has run successfully and its outputs have been inspected.

## Workflow

1. Load EEGBCI recordings for imagery runs 4, 8, and 12.
2. Standardise channel names and select EEG channels.
3. Apply a 1–40 Hz band-pass filter.
4. Read T1/T2 event annotations and create epochs from 0.5 to 4.0 seconds.
5. Prefer C3, Cz, and C4 sensorimotor channels when available.
6. Estimate Welch power spectral density and extract log band-power features for delta, theta, alpha, and beta bands.
7. Train an RBF-kernel support vector classifier using standardised features.
8. Evaluate with a single subject-independent `GroupShuffleSplit` holdout.

## Dataset and scope

- **Dataset:** PhysioNet EEG Motor Movement/Imagery (EEGBCI)
- **Runs:** 4, 8, and 12
- **Initial subjects requested:** 1–10
- **Conditions:** T1 and T2 event labels, interpreted in this script as left- and right-hand motor imagery.

The first run downloads public EEGBCI recordings, so internet access is required.

## Tools

- Python 3.10+
- MNE-Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- scikit-learn

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Project files

- [Python analysis workflow](./eeg_neurorobotics_project.py)
- [Dependency list](./requirements.txt)

## Run

From this directory:

```bash
python eeg_neurorobotics_project.py
```

## Generated outputs

The script creates `eeg_project_outputs/` with:

- `dataset_inventory.json` — participant/run loading status and event metadata
- `eeg_features.csv` — extracted epoch-level features and labels
- `example_eeg_epoch.png` — example preprocessed EEG epoch
- `mean_psd.png` — mean power spectral density
- `bandpower_by_condition.png` — alpha/beta feature summary by condition
- `confusion_matrix.png` — exploratory held-out classification results
- `heldout_predictions.csv` — held-out predictions
- `results_summary.json` — dataset and evaluation summary

Generated results are not committed as precomputed findings. Run the workflow and inspect the output files before drawing conclusions.

**Validation status:** The Python source passed a syntax compilation check. The complete data-download and analysis workflow has not yet been run end-to-end, so no empirical results are reported.

## Evaluation limitations

- The current evaluation is a **single exploratory subject-independent group holdout**.
- A single split cannot establish robust generalisation performance.
- Stronger research requires repeated group-aware validation, uncertainty estimates, per-subject reporting, and careful inspection of class balance and subject-level variability.
- The workflow does not establish real-time decoding, clinical validity, or robotic-control capability.

## Possible next steps

- Add repeated group-aware cross-validation.
- Compare simple baselines with alternative classifiers.
- Report per-subject metrics and uncertainty intervals.
- Document artefact-handling and quality-control procedures.
- Track experiment configurations for reproducibility.

## Author

**Eesha Munnangi**  
Neuro-Electrophysiology · Computational Neuroscience · Neurotechnology

---

*Exploratory learning and research code. Interpret results within the methodological limitations stated above.*
