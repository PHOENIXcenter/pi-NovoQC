"""Strict/relaxed scoring, gamma calibration and two-stage CCT filtering."""
from pathlib import Path
import json
import random
import re

import numpy as np
import pandas as pd
import torch

from .features import FEATURE_COLUMNS
from .scoring import MLPClassifier, train_mlp, gamma_fdr, calculate_fdr

PAIR = ["sample", "psm_id"]


def prepare(frame):
    frame = frame.copy()
    frame["is_decoy"] = frame.side.eq("counterpart")
    frame["denovo_label"] = frame.get("correct", pd.Series(np.nan, index=frame.index))
    frame["class"] = frame.denovo_label.where(~frame.is_decoy, False)
    # The original implementation reverses counterpart sequence tokens solely
    # as a peptide-level deduplication key. It does not generate a new peptide.
    frame["peptide_key"] = [
        "".join(reversed(re.split(r"(?<=.)(?=[A-Z])", seq))) if side else seq
        for seq, side in zip(frame.sequence, frame.is_decoy)
    ]
    return frame


def predict(frame, checkpoint, columns, device="cpu", batch_size=1024):
    # These are locally trained QC checkpoints containing a fitted scaler.
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = MLPClassifier(len(columns), dropout=0.0).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    matrix = np.nan_to_num(frame[columns].to_numpy(dtype=float), nan=0, posinf=0, neginf=0)
    matrix = state["scaler"].transform(matrix)
    scores = []
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            data = torch.as_tensor(matrix[start:start + batch_size], dtype=torch.float32, device=device)
            scores.extend(model(data).cpu().numpy())
    return np.asarray(scores)


def fit_stage(frame, folder, columns, seed=454, epochs=50, device="cpu"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    if frame.denovo_label.isna().any():
        raise ValueError("Scoring-model training requires database-reference labels")
    for offset, (name, label) in enumerate((("strict", "denovo_label"), ("relaxed", "class"))):
        random.seed(seed + offset)
        np.random.seed(seed + offset)
        torch.manual_seed(seed + offset)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + offset)
        train_mlp(frame[columns].to_numpy(), frame[label].astype(int).to_numpy(),
                  lr=0.001, dropout=0.0, batch_size=1024, epochs=epochs,
                  device=device, ckpt_path=folder / f"{name}.pt")


def score(frame, folder, columns, device="cpu"):
    result = frame.copy()
    for name in ("strict", "relaxed"):
        result[f"{name}_score"] = predict(result, Path(folder) / f"{name}.pt", columns, device)
    result["score"] = (result.strict_score + result.relaxed_score) / 2
    return result


def calibrate(frame, q):
    if frame.empty or frame.denovo_label.isna().any():
        raise ValueError("FDR calibration requires nonempty database-reference pairs")
    curve = calculate_fdr(frame, "score")
    with np.errstate(divide="ignore", invalid="ignore"):
        _, gamma, _, _, _ = gamma_fdr(curve, fdr_thresh=q)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("A finite positive gamma could not be estimated from these reference PSMs")
    return float(gamma)


def select(frame, gamma, q):
    """Select a single score prefix, including the entire cutoff score tie."""
    if frame.empty:
        return frame.copy(), frame.copy()
    curve = calculate_fdr(frame, "score")
    curve["estimated_fdr_update"] = gamma * curve.estimated_fdr
    if curve.denovo_label.isna().any():
        curve["observed_fdp"] = np.nan
    endpoints = curve.score.ne(curve.score.shift(-1))
    feasible = np.flatnonzero((endpoints & curve.estimated_fdr_update.le(q)).to_numpy())
    selected = curve.iloc[:feasible[-1] + 1].copy() if len(feasible) else curve.iloc[:0].copy()
    return curve, selected


def peptide_input(all_pairs, selected):
    """Recover both paired rows for passing PSMs, then keep the best per sequence."""
    keys = selected[PAIR].drop_duplicates()
    candidates = all_pairs.merge(keys, on=PAIR, how="inner", validate="many_to_one")
    return candidates.sort_values("score", ascending=False).drop_duplicates("peptide_key").copy()


def fit(reference, model_dir, q_psm=0.05, q_peptide=0.05, seed=454, epochs=50,
        device="cpu", acquisition="dda"):
    """Fit the four QC MLPs and two gamma factors from reference-labeled pairs."""
    if not 0 < q_psm < 1 or not 0 < q_peptide < 1:
        raise ValueError("FDR thresholds must be between zero and one")
    folder = Path(model_dir)
    folder.mkdir(parents=True, exist_ok=False)
    columns = [c for c in FEATURE_COLUMNS if acquisition != "dia" or c != "precursor_error_ppm"]
    reference = prepare(reference)
    fit_stage(reference, folder / "psm", columns, seed, epochs, device)
    psm = score(reference, folder / "psm", columns, device)
    gamma_psm = calibrate(psm, q_psm)
    curve, selected = select(psm, gamma_psm, q_psm)
    peptides = peptide_input(curve, selected)
    if peptides.empty:
        raise ValueError("No reference PSMs passed; peptide models cannot be fitted")
    fit_stage(peptides, folder / "peptide", columns, seed, epochs, device)
    peptides = score(peptides, folder / "peptide", columns, device)
    gamma_peptide = calibrate(peptides, q_peptide)
    config = dict(feature_columns=columns, q_psm=q_psm, q_peptide=q_peptide,
                  gamma_psm=gamma_psm, gamma_peptide=gamma_peptide, seed=seed,
                  acquisition=acquisition)
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def apply(frame, model_dir, output, device="cpu"):
    folder, output = Path(model_dir), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((folder / "config.json").read_text())
    columns = config["feature_columns"]
    psm = score(prepare(frame), folder / "psm", columns, device)
    curve, selected = select(psm, config["gamma_psm"], config["q_psm"])
    curve.to_csv(output / "psm_scores.csv", index=False)
    psm_targets = selected.loc[~selected.is_decoy].copy()
    psm_targets.to_csv(output / "psms.csv", index=False)
    peptides = peptide_input(curve, selected)
    peptides = score(peptides, folder / "peptide", columns, device) if len(peptides) else peptides
    pep_curve, accepted = select(peptides, config["gamma_peptide"], config["q_peptide"])
    pep_curve.to_csv(output / "peptide_scores.csv", index=False)
    targets = accepted.loc[~accepted.is_decoy].copy()
    targets.to_csv(output / "peptides.csv", index=False)
    return psm_targets, targets
