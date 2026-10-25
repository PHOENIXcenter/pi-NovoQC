"""MGF batches for the standalone GraphNovo notebook."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyteomics import mgf

from .feature_math import normalize_intensities


def mgf_batches(samples, model_config, batch_size=32, use_mgf_reference=True,
                reference_csv=None):
    references = {}
    if reference_csv:
        frame = pd.read_csv(reference_csv, dtype={"sample": str, "spectrum_id": str})
        if frame.duplicated(["sample", "spectrum_id"]).any():
            raise ValueError("Reference spectrum IDs must be unique within a sample")
        references = dict(zip(zip(frame["sample"], frame["spectrum_id"]),
                              frame["reference_sequence"]))
    spectra, precursors, metadata = [], [], []
    for sample in samples:
        paths = sorted(Path(sample["mgf_dir"]).rglob("*.mgf"))
        if not paths:
            raise FileNotFoundError(f"No MGF files in {sample['mgf_dir']}")
        for path in paths:
            with mgf.MGF(str(path)) as reader:
                for index, spectrum in enumerate(reader):
                    params = spectrum["params"]
                    charge = int(params["charge"][0])
                    if not 1 <= charge <= model_config.get("max_charge", 10):
                        continue
                    precursor_mz = float(params["pepmass"][0])
                    peaks = np.column_stack([spectrum["m/z array"], spectrum["intensity array"]])
                    peaks = peaks[np.isfinite(peaks).all(axis=1) & (peaks > 0).all(axis=1)]
                    if not len(peaks):
                        continue
                    peaks = peaks[np.argsort(peaks[:, 1])[-model_config.get("n_peaks", 150):]]
                    peaks = peaks[np.argsort(peaks[:, 0])]
                    native_id = str(params.get("title", index))
                    reference = references.get((sample["sample"], native_id))
                    if reference is None and use_mgf_reference:
                        reference = params.get("seq")
                    if pd.isna(reference):
                        reference = None
                    spectra.append(normalize_intensities(torch.as_tensor(peaks, dtype=torch.float32)))
                    precursors.append([(precursor_mz - 1.007276) * charge + 1.007276,
                                       charge, precursor_mz])
                    relative = path.relative_to(sample["mgf_dir"]).as_posix()
                    metadata.append(dict(sample=sample["sample"], psm_id=f"{relative}:{index}",
                                         spectrum_id=native_id, precursor_mz=precursor_mz,
                                         charge=charge, rt=float(params.get("rtinseconds", 0)) / 60,
                                         reference_sequence=reference))
                    if len(spectra) == batch_size:
                        yield torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True), torch.tensor(precursors), metadata
                        spectra, precursors, metadata = [], [], []
    if spectra:
        yield torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True), torch.tensor(precursors), metadata
