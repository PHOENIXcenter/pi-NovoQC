"""Paired QC features from the same inference cross-attention matrix."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import yaml

from .feature_math import (
    PeptideMass, get_theo_mz, get_aby_labels, del_wrong_premodification,
    evaluate_peptide_sequence, compute_psm_features_with_attention,
)
from .sequence_match import aa_match_batch

FEATURE_COLUMNS = [
    "aa_conf_mean", "aa_conf_var", "aa_conf_min", "aa_conf_max",
    "num_low_conf", "num_high_conf", "global_qc_score", "fragment_coverage",
    "explained_intensity", "precursor_error_ppm", "anchor_ratio",
    "aa_entropy_mean", "aa_entropy_std", "aa_entropy_min", "aa_entropy_max",
    "aa_concentration_mean", "aa_concentration_std", "aa_concentration_min",
    "aa_concentration_max", "aa_offset_mean", "aa_offset_std", "aa_offset_min",
    "aa_offset_max", "aa_center_drift_mean", "aa_center_drift_std",
    "aa_center_drift_min", "aa_center_drift_max", "aa_span_mean",
    "aa_span_std", "aa_span_min", "aa_span_max", "avg_attention_similarity",
    "fragment_alignment_consistency",
]


def paired_features(attention, sequence, mzs, intensities, precursor_mz, charge,
                    residues, top_k=10):
    """Compute target (0:L) and counterpart (1:1+L) from one matrix.

    mzs/intensities include the encoder's leading spectrum token and trailing
    precursor token. Attention columns must have the identical ordering.
    """
    length = len(re.split(r"(?<=.)(?=[A-Z])", sequence))
    if attention.shape[0] < length + 1 or attention.shape[1] != len(mzs):
        raise ValueError("Attention rows/columns do not match peptide and spectrum")
    theoretical = get_theo_mz(sequence, residues)
    ions = get_aby_labels(sequence, mzs, theoretical)
    ions = ["0"] + ions[1:-1] + ["prec"]
    masses = PeptideMass(residues=residues)
    result = []
    for start in (0, 1):
        matrix = attention[start:start + length]
        metrics = evaluate_peptide_sequence(
            matrix, ions, sequence, mzs, intensities, masses=masses,
            precursor_mass=precursor_mz, precursor_charge=charge, top_k=top_k,
        )
        result.append(compute_psm_features_with_attention(
            metrics, matrix, theoretical.reshape(-1), mzs, intensities,
        ))
    return tuple(result)


def load_model(checkpoint, config=None, device="cpu"):
    from .model import Spec2Pep
    with open(config or Path(__file__).with_name("model_config.yaml")) as handle:
        cfg = yaml.safe_load(handle)
    keys = ("dim_model", "n_head", "dim_feedforward", "n_layers", "dim_intensity",
            "max_length", "residues", "max_charge", "use_by_mask")
    model = Spec2Pep(**{key: cfg[key] for key in keys if key in cfg}, dropout=0.0)
    # Original author checkpoints also contain NumPy metadata; load trusted files only.
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = weights.get("state_dict", weights)
    weights = {key[7:] if key.startswith("module.") else key: value for key, value in weights.items()}
    # Older fine-tuning checkpoints retain auxiliary training heads that the
    # published inference model does not instantiate or call.
    auxiliary = ("aa_encoder_spectra.", "aa_encoder_peptide.", "mass_encoder.",
                 "charge_encoder.", "encoder.rel_encoder.", "encoder.rel_aa_encoder.",
                 "encoder.degree_encoder.", "encoder.edge_final.", "encoder.node_final.",
                 "encoder.final_layer_1.", "encoder.final_layer_2.",
                 "encoder.final_layer_3.", "encoder.final_layer_4.")
    weights = {key: value for key, value in weights.items() if not key.startswith(auxiliary)}
    model.load_state_dict(weights)
    return model.to(device).eval(), cfg


@torch.inference_mode()
def extract_features(model, batches, residues, device="cpu", top_k=10):
    """Yield one target/counterpart row per predicted PSM.

    Each batch contains padded spectra, precursor tensors and metadata records.
    Reference sequences are optional and are used only to assign training labels.
    """
    model.eval()
    for spectra, precursors, metadata in batches:
        spectra = spectra.to(device, dtype=torch.float32)
        precursors = precursors.to(device, dtype=torch.float32)
        predictions, _, attention_layers, _ = model(spectra, precursors)
        attention = torch.cat(attention_layers, dim=1).amax(dim=1).cpu().numpy()
        for i, meta in enumerate(metadata):
            tokens = predictions[i]
            sequence = del_wrong_premodification("".join(tokens).replace("$", ""))
            if not sequence:
                continue
            peaks = spectra[i].cpu().numpy()
            peaks = peaks[peaks[:, 0] != 0]
            # SpectrumEncoder compacts non-padding peaks before appending the precursor.
            mzs = np.r_[0, peaks[:, 0], float(precursors[i, 2])]
            intensities = np.r_[0, peaks[:, 1], 0]
            target, counterpart = paired_features(
                attention[i][:, :len(mzs)], sequence, mzs, intensities,
                float(precursors[i, 2]), int(precursors[i, 1]), residues, top_k,
            )
            reference = meta.get("reference_sequence")
            correct = None
            if reference:
                correct = bool(aa_match_batch([sequence], [reference], residues)[0][0][1])
            for side, features in (("target", target), ("counterpart", counterpart)):
                yield {**meta, "sequence": sequence, "side": side, "correct": correct,
                       **features}


def read_features(path):
    frame = pd.read_csv(path, dtype={"psm_id": str, "sample": str})
    required = {"sample", "psm_id", "sequence", "side"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if not frame.side.isin(["target", "counterpart"]).all():
        raise ValueError("side must be target or counterpart")
    keys = ["sample", "psm_id"]
    if frame.duplicated(keys + ["side"]).any() or not frame.groupby(keys).size().eq(2).all():
        raise ValueError("Every PSM must have exactly one target and one counterpart")
    if not frame.groupby(keys).sequence.nunique().eq(1).all():
        raise ValueError("Paired rows must describe the same predicted sequence")
    if "correct" not in frame:
        frame["correct"] = np.nan
    return frame
