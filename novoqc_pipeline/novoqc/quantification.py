"""MS1 peptide intensity and DB-only/DB-plus-de-novo protein quantification."""
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from pyteomics import mzml, mass

from .chromatography import process_peptide_feature, general_enzyme_digestion_with_separator
from .proteins import unmodified, inference_input, infer, target_decoy_fasta, ENZYMES
from .spectra import retention_minutes


def peptide_mz(sequence, charge):
    import re
    delta = sum(float(value) for value in re.findall(r"[+-]\d+(?:\.\d+)?", sequence))
    neutral_mass = mass.calculate_mass(sequence=unmodified(sequence)) + delta
    return neutral_mass / charge + 1.007276466621


def quantify_peptides(psms, mzml_path, output, rt_window=0.5, ppm=8, max_iso=10):
    """Quantify accepted PSMs; retention times and rt_window are in minutes."""
    if psms.empty:
        result = psms.assign(area=pd.Series(dtype=float), quantified=pd.Series(dtype=bool))
        result.to_csv(output, index=False)
        return result
    spectra, ms2 = [], {}
    requested = set(psms.spectrum_id)
    with mzml.MzML(str(mzml_path)) as reader:
        for spectrum in reader:
            rt = retention_minutes(spectrum)
            if spectrum["ms level"] == 1:
                spectrum["scanList"]["scan"][0]["scan start time"] = rt
                spectra.append(spectrum)
            elif spectrum["ms level"] == 2 and spectrum["id"] in requested:
                ms2[spectrum["id"]] = spectrum
    spectra.sort(key=lambda spec: float(spec["scanList"]["scan"][0]["scan start time"]))
    times = np.asarray([float(s["scanList"]["scan"][0]["scan start time"]) for s in spectra])
    rows = []
    for row in psms.to_dict("records"):
        spectrum = ms2[row["spectrum_id"]]
        rt = retention_minutes(spectrum)
        ion = spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]
        charge = int(ion["charge state"])
        window = spectra[np.searchsorted(times, rt - rt_window):np.searchsorted(times, rt + rt_window, side="right")]
        result = process_peptide_feature(window, unmodified(row["sequence"]),
                                         peptide_mz(row["sequence"], charge), charge, rt, None,
                                         ppm_tol=ppm, max_iso=max_iso) if window else None
        rows.append({**row, "area": np.nan, "quantified": result is not None,
                     **(result or {})})
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result


def merge_peptides(database, denovo):
    """One intensity per sample/sequence, DB preferred when both sources exist."""
    columns = ["sample", "sequence", "intensity", "msms_count", "proteins"]
    frames = []
    for source, frame in (("db", database), ("denovo", denovo)):
        if frame.empty:
            continue
        frame = frame[columns].copy()
        frame["sequence"] = frame.sequence.map(unmodified)
        if frame.intensity.lt(0).any() or frame.msms_count.lt(0).any():
            raise ValueError("Intensities and MS/MS counts must be non-negative")
        frame = frame.groupby(["sample", "sequence"], as_index=False).agg(
            intensity=("intensity", "max"), msms_count=("msms_count", "sum"),
            proteins=("proteins", lambda values: ";".join(sorted({p for v in values for p in str(v).split(";") if p}))),
        )
        frame["source"] = source
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns + ["source"])
    merged = pd.concat(frames, ignore_index=True)
    counts = merged.groupby(["sample", "sequence"]).msms_count.sum()
    merged = merged.drop_duplicates(["sample", "sequence"], keep="first")
    merged["msms_count"] = [counts.loc[(s, p)] for s, p in zip(merged["sample"], merged.sequence)]
    return merged


def theoretical_counts(fasta, enzyme="Trypsin", missed_cleavages=2):
    result = {}
    for record in SeqIO.parse(fasta, "fasta"):
        fragments = general_enzyme_digestion_with_separator(str(record.seq), enzyme).split(" | ")
        peptides = {"".join(fragments[i:j]) for i in range(len(fragments))
                    for j in range(i + 1, min(len(fragments), i + missed_cleavages + 1) + 1)}
        result[record.id] = sum(6 <= len(peptide) <= 40 for peptide in peptides)
    return result


def quantify_proteins(database, denovo, fasta, output, enzyme="Trypsin", top_n=3):
    """Re-infer groups using intensity scores without a new protein-FDR filter."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    theoretical = theoretical_counts(fasta, enzyme)
    database = database.copy()
    combined_fasta = output / "target_decoy.fasta"
    target_decoy_fasta(fasta, combined_fasta)
    digest_rule = ENZYMES[enzyme]
    summaries = []
    for scheme, added in (("db_only", denovo.iloc[:0]), ("db_plus_denovo", denovo)):
        merged = merge_peptides(database, added)
        merged.to_csv(output / f"{scheme}_peptides.csv", index=False)
        for index, (sample, group) in enumerate(merged.groupby("sample", sort=True)):
            folder = output / f"{scheme}_{index}"
            folder.mkdir()
            mapping = group[["sequence", "proteins"]].assign(protein=lambda x: x.proteins.str.split(";")).explode("protein")
            mapping["score"] = 1.0
            input_path = folder / "peptide_intensities.tsv"
            inference_input(mapping, input_path, dict(zip(group.sequence, group.intensity)))
            if not group.intensity.gt(0).any():
                continue
            proteins = infer(input_path, folder / "inference", q=1.0, filter_q=False,
                             enzyme=digest_rule, database=combined_fasta)
            joined = proteins.merge(group, left_on="Peptides", right_on="sequence", how="inner", validate="many_to_one")
            for protein, values in joined.groupby("Protein"):
                count = theoretical.get(protein)
                if count is None:
                    raise ValueError(f"Protein {protein} is absent from the supplied FASTA")
                total_msms = values.msms_count.sum()
                summaries.append(dict(
                    sample=sample, scheme=scheme, protein=protein,
                    peptide_count=values.sequence.nunique(), theoretical_peptides=count,
                    peptide_count_coverage=values.sequence.nunique() / count if count else np.nan,
                    ibaq=float(values.Score.iloc[0]) / count if count else np.nan,
                    top_n_mean=values.intensity.nlargest(top_n).mean(),
                    weighted_mean=(values.intensity * values.msms_count).sum() / total_msms if total_msms else np.nan,
                    msms_count=total_msms,
                ))
    result = pd.DataFrame(summaries)
    result.to_csv(output / "protein_quantification.csv", index=False)
    return result
