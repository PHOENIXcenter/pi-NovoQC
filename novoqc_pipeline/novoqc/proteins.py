"""BLAST mapping and separate target-decoy protein inference."""
from pathlib import Path
import csv
import re
import subprocess

import pandas as pd
from Bio import SeqIO
import yaml

ENZYMES = {"Trypsin": "trypsin", "AspN": "asp-n", "LysN": "lysn",
           "LysC": "lysc", "GluC": "glutamyl endopeptidase",
           "Chymotrypsin": "chymotrypsin high specificity"}


def unmodified(sequence):
    return "".join(re.findall("[A-Z]", sequence))


def target_decoy_fasta(source, destination, prefix="DECOY_"):
    records = list(SeqIO.parse(source, "fasta"))
    if any(record.id.startswith(prefix) for record in records):
        raise ValueError("Provide a target-only FASTA; reverse decoys are generated here")
    with open(destination, "x") as handle:
        for record in records:
            handle.write(f">{record.id}\n{record.seq}\n>{prefix}{record.id}\n{str(record.seq)[::-1]}\n")


def map_peptides(peptides, fasta, output, evalue=0.1, min_length=6, threads=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sequences = sorted({unmodified(seq) for seq in peptides.sequence if len(unmodified(seq)) >= min_length})
    query = output / "peptides.fasta"
    query.write_text("".join(f">p{i}\n{seq}\n" for i, seq in enumerate(sequences)))
    database = output / "target_decoy.fasta"
    target_decoy_fasta(fasta, database)
    subprocess.run(["makeblastdb", "-in", str(database), "-dbtype", "prot",
                    "-parse_seqids", "-out", str(output / "proteins")], check=True)
    hits = output / "matches.tsv"
    if not sequences:
        return pd.DataFrame(columns=["sequence", "protein", "score"])
    subprocess.run(["blastp", "-task", "blastp-short", "-query", str(query), "-db", str(output / "proteins"),
                    "-out", str(hits), "-evalue", str(evalue), "-num_threads", str(threads),
                    "-outfmt", "6 qseqid sseqid length mismatch gapopen qlen qstart qend score"], check=True)
    columns = ["query", "protein", "length", "mismatch", "gapopen", "qlen", "start", "end", "score"]
    if not hits.stat().st_size:
        return pd.DataFrame(columns=["sequence", "protein", "score"])
    frame = pd.read_csv(hits, sep="\t", names=columns)
    frame = frame.loc[(frame.length == frame.qlen) & (frame.mismatch == 0) & (frame.gapopen == 0)
                      & (frame.start == 1) & (frame.end == frame.qlen)].copy()
    frame["sequence"] = frame["query"].map({f"p{i}": seq for i, seq in enumerate(sequences)})
    return frame[["sequence", "protein", "score"]].drop_duplicates()


def inference_input(mapping, output, intensities=None):
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["PSMId", "Score", "peptide", "proteinIds"])
        for i, (sequence, group) in enumerate(mapping.groupby("sequence", sort=True)):
            value = group.score.max() if intensities is None else intensities.get(sequence, 0)
            if value > 0:
                writer.writerow([str(i), float(value), sequence, *sorted(group.protein.unique())])


def infer(input_path, output, q=0.01, enzyme="trypsin", prefix="DECOY_", filter_q=True,
          database=None):
    from pyproteininference.parameters import ProteinInferenceParameter
    from pyproteininference.in_silico_digest import PyteomicsDigest
    from pyproteininference.reader import GenericReader
    from pyproteininference.datastore import DataStore
    from pyproteininference.scoring import Score
    from pyproteininference.inference import Inference
    from pyproteininference.export import Export
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = {"parameters": {
        "general": dict(export="long", fdr=q, picker=True, tag="novoqc"),
        "data_restriction": dict(pep_restriction="None", peptide_length_restriction="None",
                                 q_value_restriction="None", custom_restriction="None"),
        "score": dict(protein_score="additive", psm_score="Score", psm_score_type="additive"),
        "identifiers": dict(decoy_symbol=prefix, isoform_symbol="", reviewed_identifier_symbol=""),
        "inference": dict(inference_type="parsimony", grouping_type="subset_peptides"),
        "digest": dict(digest_type=enzyme, missed_cleavages=3),
        "parsimony": dict(lp_solver="pulp", shared_peptides="best"),
        "peptide_centric": dict(max_identifiers=5),
    }}
    params = output / "parameters.yaml"
    params.write_text(yaml.safe_dump(config, sort_keys=False))
    result_path = output / "all_protein_groups.csv"
    settings = ProteinInferenceParameter(yaml_param_filepath=str(params))
    digest = PyteomicsDigest(database_path=str(database) if database else None,
                            digest_type=settings.digest_type,
                            missed_cleavages=settings.missed_cleavages,
                            reviewed_identifier_symbol=settings.reviewed_identifier_symbol,
                            max_peptide_length=settings.restrict_peptide_length, id_splitting=False)
    if database:
        digest.digest_fasta_database()
    reader = GenericReader(combined_files=[str(input_path)], parameter_file_object=settings,
                           digest=digest, append_alt_from_db=False)
    reader.read_psms()
    data = DataStore(reader=reader, digest=digest)
    data.restrict_psm_data()
    data.recover_mapping()
    data.create_scoring_input()
    Score(data=data).score_psms(score_method=settings.protein_score)
    # Additive scores are higher-is-better even for a single protein or tied scores.
    data.high_low_better = data.HIGHER_PSM_SCORE
    data.protein_picker()
    Inference.run_inference(data=data, digest=digest)
    data.calculate_q_values()
    Export(data=data).export_to_csv(output_filename=str(result_path), export_type="long")
    result = pd.read_csv(result_path)
    keep = ~result.Protein.astype(str).str.startswith(prefix)
    if filter_q:
        keep &= result.Q_Value.le(q)
    result = result.loc[keep].copy()
    result.to_csv(output / "proteins.csv", index=False)
    return result


def protein_qc(peptides, fasta, output, q=0.01, threads=1, enzyme="trypsin"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mapping = map_peptides(peptides, fasta, output / "mapping", threads=threads)
    mapping.to_csv(output / "peptide_protein_matches.csv", index=False)
    if mapping.empty:
        empty = pd.DataFrame(columns=["Protein", "Score", "Q_Value", "Peptides"])
        empty.to_csv(output / "proteins.csv", index=False)
        return empty, mapping
    input_path = output / "inference_input.tsv"
    inference_input(mapping, input_path)
    proteins = infer(input_path, output / "inference", q=q, enzyme=enzyme,
                     database=output / "mapping/target_decoy.fasta")
    proteins.to_csv(output / "proteins.csv", index=False)
    return proteins, mapping
