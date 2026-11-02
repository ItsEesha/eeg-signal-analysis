"""
EEG Signal Analysis for Neural Feature Extraction
Dataset: PhysioNet EEG Motor Movement/Imagery (EEGBCI)

Run: python eeg_neurorobotics_project.py
Requires internet on first run to download public data.
This is offline research code, not a clinical tool or real-time robot controller.
"""
from pathlib import Path
import json
import warnings

from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.time_frequency import psd_array_welch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)

OUT = Path("eeg_project_outputs")
OUT.mkdir(exist_ok=True)
SUBJECTS = list(range(1, 11))  # Start small; expand after checking outputs.
RUNS = [4, 8, 12]              # Motor-imagery runs in EEGBCI.
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
SFREQ_FILTER = (1.0, 40.0)
TMIN, TMAX = 0.5, 4.0
SEED = 42


def extract_bandpower(epochs):
    """Log integrated Welch PSD for each epoch, channel, and frequency band."""
    data = epochs.get_data(copy=True)
    sfreq = float(epochs.info["sfreq"])
    psd, freqs = psd_array_welch(
        data, sfreq=sfreq, fmin=1, fmax=30,
        n_per_seg=min(int(2 * sfreq), data.shape[-1]),
        n_overlap=0, average="mean", verbose=False
    )
    blocks, columns = [], []
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            raise ValueError(f"No PSD bins for {band}")
        power = trapezoid(psd[:, :, mask], x=freqs[mask], axis=-1)
        blocks.append(np.log10(np.maximum(power, np.finfo(float).tiny)))
        columns.extend([f"{ch}_{band}_logpower" for ch in epochs.ch_names])
    return np.concatenate(blocks, axis=1), columns


def load_subject(subject):
    """Load and preprocess imagery runs; keep T1/T2 epochs if both labels exist."""
    inventory = {"subject": subject, "runs": RUNS, "loaded_runs": [], "events": [], "status": "started"}
    try:
        files = eegbci.load_data(subject, RUNS, update_path=True, verbose=False)
    except Exception as exc:
        inventory["status"] = f"download_error: {exc}"
        warnings.warn(f"Subject {subject}: {exc}")
        return None, inventory

    all_epochs = []
    for run, fname in zip(RUNS, files):
        try:
            raw = mne.io.read_raw_edf(fname, preload=True, verbose="ERROR")
            eegbci.standardize(raw)
            raw.pick("eeg")
            raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
            raw.filter(SFREQ_FILTER[0], SFREQ_FILTER[1], verbose="ERROR")
            events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
            inventory["loaded_runs"].append(run)
            inventory["events"].append({"run": run, "event_id": {k: int(v) for k, v in event_id.items()}})
            selected = {k: event_id[k] for k in ("T1", "T2") if k in event_id}
            if set(selected) != {"T1", "T2"}:
                warnings.warn(f"Subject {subject}, run {run}: T1/T2 missing; events={event_id}")
                continue
            ep = mne.Epochs(
                raw, events, event_id=selected, tmin=TMIN, tmax=TMAX,
                baseline=None, preload=True, reject_by_annotation=True,
                event_repeated="drop", verbose="ERROR"
            )
            # Prefer central sensorimotor channels if available.
            preferred = [ch for ch in ("C3", "Cz", "C4") if ch in ep.ch_names]
            ep.pick(preferred if len(preferred) >= 2 else ep.ch_names)
            if len(ep):
                all_epochs.append(ep)
        except Exception as exc:
            warnings.warn(f"Subject {subject}, run {run}: processing error: {exc}")

    if not all_epochs:
        inventory["status"] = "no_usable_epochs"
        return None, inventory
    try:
        epochs = mne.concatenate_epochs(all_epochs, verbose="ERROR")
        inventory["status"] = "ok"
        inventory["epochs_kept"] = len(epochs)
        return epochs, inventory
    except Exception as exc:
        inventory["status"] = f"concatenation_error: {exc}"
        return None, inventory


def main():
    X_all, y_all, group_all, inventory, first_epochs = [], [], [], [], None
    feature_names = None

    for subject in SUBJECTS:
        print(f"Processing subject {subject}...")
        epochs, meta = load_subject(subject)
        inventory.append(meta)
        if epochs is None or len(epochs) == 0:
            continue

        code_to_name = {v: k for k, v in epochs.event_id.items()}
        names = np.array([code_to_name.get(code, "unknown") for code in epochs.events[:, -1]])
        keep = np.isin(names, ["T1", "T2"])
        epochs = epochs[keep]
        names = names[keep]
        y = np.where(names == "T1", 0, 1)  # In these imagery runs: T1 left, T2 right.
        if len(np.unique(y)) != 2:
            warnings.warn(f"Subject {subject}: only one class; skipping.")
            continue

        # Consistent channel set is required across subjects.
        if feature_names is not None and epochs.ch_names != first_epochs.ch_names:
            warnings.warn(f"Subject {subject}: channel set differs; skipping for consistency.")
            continue
        X, cols = extract_bandpower(epochs)
        if feature_names is None:
            feature_names = cols
            first_epochs = epochs.copy()
        X_all.append(X)
        y_all.append(y)
        group_all.append(np.full(len(y), subject))

    pd.DataFrame(inventory).to_json(OUT / "dataset_inventory.json", orient="records", indent=2)
    if not X_all:
        raise RuntimeError(
            "No usable epochs found. Check internet access, MNE/PhysioNet download, "
            "event annotations, and installed package versions."
        )

    X = np.vstack(X_all)
    y = np.concatenate(y_all).astype(int)
    groups = np.concatenate(group_all).astype(int)

    feat = pd.DataFrame(X, columns=feature_names)
    feat.insert(0, "subject", groups)
    feat.insert(1, "label_code", y)
    feat.insert(2, "label", np.where(y == 0, "T1_left_imagery", "T2_right_imagery"))
    feat.to_csv(OUT / "eeg_features.csv", index=False)

    print(f"Epochs: {len(y)} | Subjects: {len(np.unique(groups))} | Features: {X.shape[1]}")
    print("Class counts:", pd.Series(y).value_counts().sort_index().to_dict())

    # Example epoch trace.
    try:
        data = first_epochs.get_data(copy=True)
        times = first_epochs.times
        plt.figure(figsize=(10, 5))
        for idx, ch in enumerate(first_epochs.ch_names):
            plt.plot(times, data[0, idx] * 1e6 + idx * 100, label=ch)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV; vertically offset)")
        plt.title("Example preprocessed EEG epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / "example_eeg_epoch.png", dpi=180)
        plt.close()
    except Exception as exc:
        warnings.warn(f"Epoch plot failed: {exc}")

    # Mean PSD plot.
    try:
        data = first_epochs.get_data(copy=True)
        psd, freqs = psd_array_welch(
            data, sfreq=float(first_epochs.info["sfreq"]), fmin=1, fmax=40,
            n_per_seg=min(int(2 * first_epochs.info["sfreq"]), data.shape[-1]),
            verbose=False
        )
        mean_psd = psd.mean(axis=(0, 1))
        plt.figure(figsize=(8, 5))
        plt.plot(freqs, 10 * np.log10(np.maximum(mean_psd, np.finfo(float).tiny)))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (dB, relative units)")
        plt.title("Mean EEG power spectral density")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / "mean_psd.png", dpi=180)
        plt.close()
    except Exception as exc:
        warnings.warn(f"PSD plot failed: {exc}")

    # Alpha/beta feature summary by condition.
    band_cols = [c for c in feature_names if c.endswith("_alpha_logpower") or c.endswith("_beta_logpower")]
    if band_cols:
        summary = feat.groupby("label")[band_cols].mean().T
        summary.plot(kind="bar", figsize=(10, 5))
        plt.ylabel("Mean log10 band power")
        plt.title("Alpha/beta log-power features by condition")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(OUT / "bandpower_by_condition.png", dpi=180)
        plt.close()

    result = {
        "dataset": "PhysioNet EEG Motor Movement/Imagery Dataset",
        "runs": RUNS,
        "subjects_requested": SUBJECTS,
        "subjects_included": np.unique(groups).tolist(),
        "n_epochs": int(len(y)),
        "n_features": int(X.shape[1]),
        "class_counts": {"T1_left_imagery": int((y == 0).sum()), "T2_right_imagery": int((y == 1).sum())},
    }

    # Subject-independent holdout: participants in test set never occur in training set.
    if len(np.unique(groups)) >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train, g_test = groups[train_idx], groups[test_idx]

        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        result["evaluation"] = {
            "method": "single subject-independent GroupShuffleSplit holdout (exploratory)",
            "train_subjects": np.unique(g_train).tolist(),
            "test_subjects": np.unique(g_test).tolist(),
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "classification_report": classification_report(
                y_test, pred, labels=[0, 1],
                target_names=["T1_left_imagery", "T2_right_imagery"],
                output_dict=True, zero_division=0
            ),
        }
        if len(np.unique(y_test)) == 2:
            result["evaluation"]["roc_auc"] = float(roc_auc_score(y_test, prob))

        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["T1 left imagery", "T2 right imagery"]).plot(values_format="d")
        plt.title("Subject-independent SVM evaluation")
        plt.tight_layout()
        plt.savefig(OUT / "confusion_matrix.png", dpi=180)
        plt.close()

        pd.DataFrame({
            "subject": g_test,
            "true_label": np.where(y_test == 0, "T1_left_imagery", "T2_right_imagery"),
            "predicted_label": np.where(pred == 0, "T1_left_imagery", "T2_right_imagery"),
        }).to_csv(OUT / "heldout_predictions.csv", index=False)
        print("Accuracy:", round(result["evaluation"]["accuracy"], 3))
        print("Balanced accuracy:", round(result["evaluation"]["balanced_accuracy"], 3))
    else:
        result["evaluation"] = {"status": "Not enough distinct participants for subject-independent holdout."}

    (OUT / "results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Finished. Outputs saved in: {OUT.resolve()}")
    print("Review dataset_inventory.json and results_summary.json before writing conclusions.")


if __name__ == "__main__":
    main()
