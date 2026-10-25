import argparse
from pathlib import Path

import pandas as pd
from .proteins import ENZYMES


def main():
    parser = argparse.ArgumentParser(prog="novoqc", description="NovoQC identification and quantification")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one mzML sample through all stages")
    extract = commands.add_parser("features", help="Predict peptides and extract paired QC features")
    for sub in (run, extract):
        sub.add_argument("--mzml", required=True)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--references", required=True, help="Reference CSV: sample,spectrum_id,reference_sequence")
        sub.add_argument("--sample", required=True)
        sub.add_argument("--model-config")
        sub.add_argument("--batch-size", type=int, default=128)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--output", required=True)
    run.add_argument("--fasta", required=True, help="Target-only protein FASTA")
    run.add_argument("--database-quant", required=True, help="Database peptide intensities CSV")
    run.add_argument("--protein-q", type=float, default=0.01)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--enzyme", choices=ENZYMES, default="Trypsin")
    fit = commands.add_parser("fit", help="Train PSM/peptide scoring models and estimate gamma")
    fit.add_argument("--features", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--acquisition", choices=["dda", "dia"], default="dda")
    for sub in (run, fit):
        sub.add_argument("--psm-q", type=float, default=0.05)
        sub.add_argument("--peptide-q", type=float, default=0.05)
        sub.add_argument("--seed", type=int, default=454)
        sub.add_argument("--epochs", type=int, default=50)
    apply = commands.add_parser("qc", help="Apply fitted PSM/peptide QC")
    apply.add_argument("--features", required=True)
    apply.add_argument("--models", required=True)
    apply.add_argument("--output", required=True)
    apply.add_argument("--device", default="cpu")
    protein = commands.add_parser("proteins", help="Protein mapping, inference and q-value filtering")
    protein.add_argument("--peptides", required=True)
    protein.add_argument("--fasta", required=True)
    protein.add_argument("--output", required=True)
    protein.add_argument("--q", type=float, default=0.01)
    protein.add_argument("--threads", type=int, default=1)
    protein.add_argument("--enzyme", choices=ENZYMES, default="Trypsin")
    quant = commands.add_parser("quantify", help="Quantify accepted PSMs from MS1 chromatograms")
    quant.add_argument("--psms", required=True)
    quant.add_argument("--mzml", required=True)
    quant.add_argument("--output", required=True)
    pq = commands.add_parser("protein-quant", help="Merge peptide intensities and re-infer protein groups")
    pq.add_argument("--database", required=True)
    pq.add_argument("--denovo", required=True)
    pq.add_argument("--fasta", required=True)
    pq.add_argument("--enzyme", choices=ENZYMES, default="Trypsin")
    pq.add_argument("--output", required=True)
    args = parser.parse_args()
    from . import features, qc, proteins, quantification

    if args.command in ("features", "run"):
        from .spectra import batches
        output = Path(args.output)
        if args.command == "run":
            output.mkdir(parents=True, exist_ok=False)
            feature_path = output / "features.csv"
        else:
            feature_path = output
            if feature_path.exists():
                raise FileExistsError(feature_path)
        references = pd.read_csv(args.references, dtype={"sample": str, "spectrum_id": str})
        model, config = features.load_model(args.checkpoint, args.model_config, args.device)
        data = batches(args.mzml, references, args.sample, args.batch_size,
                       config.get("n_peaks", 150), config.get("max_charge", 10))
        frame = pd.DataFrame(features.extract_features(model, data, config["residues"], args.device))
        frame.to_csv(feature_path, index=False)
        del model
        if args.command == "features":
            return
        labeled = frame.loc[frame.correct.notna()].copy()
        qc.fit(labeled, output / "models", args.psm_q, args.peptide_q,
               args.seed, args.epochs, args.device)
        psms, peptides = qc.apply(frame, output / "models", output / "qc", args.device)
        # Database-search-unidentified spectra supply additional protein evidence.
        additional = peptides.loc[peptides.reference_sequence.isna()].copy()
        accepted, mapping = proteins.protein_qc(additional, args.fasta, output / "protein_qc",
                                                q=args.protein_q, threads=args.threads,
                                                enzyme=ENZYMES[args.enzyme])
        psms = psms.loc[psms.reference_sequence.isna() & psms.sequence.isin(additional.sequence)].copy()
        psms["unmodified_sequence"] = psms.sequence.map(proteins.unmodified)
        psms = psms.loc[psms.unmodified_sequence.isin(accepted.Peptides)]
        peptide_quant = quantification.quantify_peptides(psms, args.mzml, output / "peptide_quantification.csv")
        denovo = pd.DataFrame(columns=["sample", "sequence", "intensity", "msms_count", "proteins"])
        if len(peptide_quant):
            denovo = peptide_quant.loc[peptide_quant.quantified].groupby(["sample", "unmodified_sequence"], as_index=False).agg(
                intensity=("area", "max"), msms_count=("psm_id", "nunique"))
            denovo = denovo.rename(columns={"unmodified_sequence": "sequence"})
            accepted_mapping = accepted.groupby("Peptides").Protein.agg(lambda x: ";".join(sorted(set(x))))
            denovo["proteins"] = denovo.sequence.map(accepted_mapping)
        denovo.to_csv(output / "denovo_peptide_intensities.csv", index=False)
        database = pd.read_csv(args.database_quant)
        database = database.loc[database["sample"].eq(args.sample)]
        quantification.quantify_proteins(database, denovo, args.fasta, output / "quantification", args.enzyme)
    elif args.command == "fit":
        qc.fit(features.read_features(args.features), args.output, args.psm_q, args.peptide_q,
               args.seed, args.epochs, args.device, args.acquisition)
    elif args.command == "qc":
        qc.apply(features.read_features(args.features), args.models, args.output, args.device)
    elif args.command == "proteins":
        proteins.protein_qc(pd.read_csv(args.peptides), args.fasta, args.output, args.q, args.threads,
                           ENZYMES[args.enzyme])
    elif args.command == "quantify":
        quantification.quantify_peptides(pd.read_csv(args.psms), args.mzml, args.output)
    elif args.command == "protein-quant":
        quantification.quantify_proteins(pd.read_csv(args.database), pd.read_csv(args.denovo),
                                       args.fasta, args.output, args.enzyme)


if __name__ == "__main__":
    main()
