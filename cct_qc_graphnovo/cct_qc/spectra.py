"""mzML input using native spectrum identifiers."""
from pathlib import Path
import numpy as np
import torch
from pyteomics import mzml

from .feature_math import normalize_intensities


def retention_minutes(spectrum):
    value = spectrum["scanList"]["scan"][0]["scan start time"]
    unit = getattr(value, "unit_info", "minute")
    return float(value) / 60 if unit == "second" else float(value)


def batches(path, references=None, sample=None, batch_size=128, n_peaks=150,
            max_charge=10):
    sample = sample or Path(path).stem
    reference_map = {}
    if references is not None:
        rows = references.loc[references["sample"].eq(sample)]
        if rows.spectrum_id.duplicated().any():
            raise ValueError("Reference table contains duplicate spectrum IDs within a sample")
        reference_map = dict(zip(rows.spectrum_id, rows.reference_sequence))
    peaks_batch, precursor_batch, metadata = [], [], []
    with mzml.MzML(str(path)) as reader:
        for spectrum in reader:
            if spectrum["ms level"] != 2:
                continue
            ion = spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]
            charge = int(ion.get("charge state", 0))
            if not 1 <= charge <= max_charge:
                continue
            mz = float(ion["selected ion m/z"])
            peaks = np.column_stack([spectrum["m/z array"], spectrum["intensity array"]])
            peaks = peaks[np.isfinite(peaks).all(axis=1) & (peaks[:, 0] > 0) & (peaks[:, 1] > 0)]
            if not len(peaks):
                continue
            peaks = peaks[np.argsort(peaks[:, 1])[-n_peaks:]]
            peaks = peaks[np.argsort(peaks[:, 0])]
            tensor = torch.as_tensor(peaks, dtype=torch.float32)
            tensor = normalize_intensities(tensor)
            native_id = spectrum["id"]
            peaks_batch.append(tensor)
            # Match the precursor representation expected by the saved model.
            precursor_batch.append([(mz - 1.007276) * charge + 1.007276, charge, mz])
            metadata.append(dict(sample=sample, psm_id=native_id, spectrum_id=native_id,
                                 precursor_mz=mz, charge=charge, rt=retention_minutes(spectrum),
                                 reference_sequence=reference_map.get(native_id)))
            if len(peaks_batch) == batch_size:
                yield torch.nn.utils.rnn.pad_sequence(peaks_batch, batch_first=True), torch.tensor(precursor_batch), metadata
                peaks_batch, precursor_batch, metadata = [], [], []
    if peaks_batch:
        yield torch.nn.utils.rnn.pad_sequence(peaks_batch, batch_first=True), torch.tensor(precursor_batch), metadata
