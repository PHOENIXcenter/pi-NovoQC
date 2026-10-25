# π-NovoQC: from DDA spectra to protein quantification

This workflow accompanies **False discovery rate control enables trustworthy
deep-learning-based de novo peptide sequencing** (Liang et al.). It implements
the analysis in Fig. 3c: π-xNovo-DDA prediction, CCT at the PSM and peptide
levels, protein inference and MS1 quantification. Feature definitions are in
Supplementary Table 2 and Supplementary Note 2. See `CITATION.cff` for citation.

NovoQC combines paired feature construction, CCT quality control at the PSM
and peptide levels, separate target-decoy protein inference, and MS1-based
peptide and protein quantification.

This directory contains its own xNovo-DDA inference model, checkpoint loader,
paired feature code and required depthcharge fork under `novoqc/`. It can be
copied and installed independently and does not import either xNovo training
directory. The xNovo sequencing checkpoint is fixed during analysis; the QC
scoring MLPs are trained separately using database-reference labels.

The revised feature construction uses the **same inference cross-attention
matrix** for target and counterpart features. For a peptide with length `L`,
the row slices are `0:L` and `1:1+L`. There is no additional model call for
counterpart feature extraction. Autoregressive peptide prediction itself still
requires successive decoding steps.

## Installation

Use a dedicated environment; Python 3.12 is used in the example below. Install a PyTorch build
appropriate for your CPU or CUDA version, then install this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Protein mapping also requires NCBI BLAST+ (`makeblastdb` and `blastp`) on `PATH`.
PyProteinInference performs parsimonious protein inference using PuLP/CBC.
The existing sequencing-model fine-tuning projects remain separate; this
package loads a trained checkpoint and does not retrain the sequencing model.

## Complete workflow for one DDA sample

Run from this directory. Prepare the five input files below before launching:
the mzML must retain MS1 scans for quantification; the reference PSMs and database
peptide intensities come from your completed database-search workflow. Use a
target-only FASTA containing the protein IDs in the quantitative table.

Download the DDA pretrained models from
[`pretrained_model_for_DDA.zip`](https://drive.google.com/file/d/1YLYgdNoA5B12j7xW-HM-2oapZlQsHZ8L/view).
Set `--checkpoint` to the extracted xNovo-DDA checkpoint and use its matching
model configuration. Replace `pi_xnovo_dda.pth` in the example with that path.

```bash
novoqc run \
  --mzml sample.mzML \
  --sample sample \
  --references reference_psms.csv \
  --checkpoint pi_xnovo_dda.pth \
  --fasta target_proteins.fasta \
  --database-quant database_peptide_intensities.csv \
  --device cuda:0 \
  --psm-q 0.05 --peptide-q 0.05 --protein-q 0.01 \
  --enzyme Trypsin \
  --output results/sample
```

The output directory must be new. Database-search reference PSMs are used to
train the strict and relaxed scoring models and estimate the FDR adjustment
factors. Database searches and their upstream QC are performed before this
command. Provide only reference identifications that passed your search QC.

`reference_psms.csv` contains:

| Column | Meaning |
| --- | --- |
| `sample` | Sample name matching `--sample` |
| `spectrum_id` | Exact mzML native spectrum ID, not its position in the file |
| `reference_sequence` | Database-search peptide in the checkpoint's modification notation |

`database_peptide_intensities.csv` contains:

| Column | Meaning |
| --- | --- |
| `sample` | Sample name |
| `sequence` | Peptide sequence |
| `intensity` | Database-workflow peptide intensity |
| `msms_count` | Number of supporting MS/MS observations |
| `proteins` | Semicolon-separated protein IDs matching the target FASTA |

Format examples only, to be replaced with your qualified data:

```csv
sample,spectrum_id,reference_sequence
sample,controllerType=0 controllerNumber=1 scan=1001,PEPTIDER
```

```csv
sample,sequence,intensity,msms_count,proteins
sample,PEPTIDER,125000.0,3,P12345
```

The native spectrum ID must match the mzML exactly; converting a scan number to
an ID by guessing its prefix can prevent reference matches. Keep one qualified
reference sequence per spectrum. Peptide modifications must use the checkpoint's
token notation, not the original search tool's notation if it differs.

The shipped model configuration matches the original pi-xNovo DDA architecture.
Use `--model-config` when the checkpoint architecture or residue vocabulary
differs. Modification tokens include `C+57.021`, `M+15.995` and an N-terminal
`+42.011`; residue masses must match those used to train the checkpoint.
Only load trusted sequencing and QC checkpoints: the original checkpoint
format includes Python-serialized metadata and fitted scalers.

The workflow writes paired features, fitted QC models, all scored PSMs/peptides,
accepted target PSMs/peptides, protein mappings and protein groups, peptide
intensities, and DB-only/DB-plus-de-novo protein quantitative results. Only
peptides from database-search-unidentified spectra that pass peptide and protein
QC contribute additional de novo protein evidence. Protein inference is rerun
after combining quantitative evidence, without another protein-FDR filter.

| Output within `results/sample/` | Contents |
| --- | --- |
| `features.csv` | Paired target/counterpart features for predicted PSMs |
| `models/` | Strict/relaxed scoring models for both QC levels and FDR adjustment factors |
| `qc/psms.csv`, `qc/peptides.csv` | Accepted target PSMs and peptides |
| `qc/psm_scores.csv`, `qc/peptide_scores.csv` | Score tables and FDR estimates |
| `protein_qc/peptide_protein_matches.csv`, `protein_qc/proteins.csv` | Additional peptide mappings and protein-QC results |
| `peptide_quantification.csv` | MS1 extraction results; inspect `quantified` and `area` |
| `denovo_peptide_intensities.csv` | Additional peptide intensities prepared for protein quantification |
| `quantification/protein_quantification.csv` | Protein quantitative results for database-only and combined evidence |

## Run stages separately

To stop at PSM/peptide-level QC, run only `features`, `fit` and `qc` below.
Their output includes `psms.csv` and `peptides.csv`. Protein mapping, protein
FDR filtering and quantification are optional downstream steps; peptide outputs
can instead be used in SAP analysis or other peptide-level studies.

```bash
novoqc features --mzml sample.mzML --sample sample \
  --references reference_psms.csv --checkpoint pi_xnovo_dda.pth \
  --device cuda:0 --output features.csv

python -c 'import pandas as pd; f = pd.read_csv("features.csv"); f.loc[f.correct.notna()].to_csv("reference_features.csv", index=False)'

novoqc fit --features reference_features.csv --device cuda:0 \
  --psm-q 0.05 --peptide-q 0.05 --output models
novoqc qc --features features.csv --models models --device cuda:0 --output qc_results
```

You can now use `qc_results/psms.csv` and `qc_results/peptides.csv` without
performing protein analysis. The filtering thresholds are saved during `fit`;
to change them, fit into a new model directory at the requested thresholds.

For the integrated protein/quantification analysis, use `novoqc run` above. It
selects additional evidence from database-search-unidentified spectra and passes
the appropriate accepted PSMs and peptide-protein mappings between stages.
The lower-level `proteins`, `quantify` and `protein-quant` commands are available
for custom workflows; inspect each with `novoqc COMMAND --help` before using it.

For `fit`, retain only rows with database-reference labels. Feature CSVs have one
`target` and one `counterpart` row per `(sample, psm_id)`, the same predicted
`sequence` in both rows, and the feature columns listed in
`novoqc/features.py`. `correct` is the database-reference match label shared by
both paired rows; it is empty for spectra without a reference identification.
Do not label unknown predictions as incorrect.

`fit --acquisition dia` accepts the corresponding 32-feature DIA tables and
omits precursor mass error. The included mzML-to-prediction adapter is for
pi-xNovo DDA; DIA or other architectures must supply their own paired feature
tables. Non-attention models require a separately designed feature extractor.

All counterpart rows contribute to the CCT count estimate. The two MLP scores
are averaged, gamma is estimated on reference data, and filtering uses one score
threshold with the entire cutoff score tie included. Peptide QC recovers paired
rows from accepted PSMs, deduplicates by the original sequence-key convention,
then applies separately fitted peptide-level models and gamma. Gamma values are
specific to the target FDR thresholds used during fitting.

Peptide intensity follows the original isotope-XIC calculation. Protein outputs
include observed/theoretical peptide counts, peptide-count coverage, iBAQ, mean
Top3 intensity, MS/MS-count-weighted mean intensity and total MS/MS count.
Missing MS1 signals remain missing, rather than being interpreted as measured
zero intensity. The theoretical digestion settings use up to two missed
cleavages and peptides of length 6-40.

## Python API

The functions in `novoqc.features`, `novoqc.qc`, `novoqc.proteins` and
`novoqc.quantification` accept arrays or pandas tables. For example:

```python
from novoqc.features import read_features
from novoqc.qc import fit, apply

frame = read_features("features.csv")
reference = frame.loc[frame.correct.notna()].copy()
fit(reference, "models", q_psm=0.05, q_peptide=0.05, device="cuda:0")
psms, peptides = apply(frame, "models", "results", device="cuda:0")
```

## Common setup issues

- `makeblastdb` and `blastp` must be installed separately and available on `PATH`
  for protein analysis; they are not required for the three QC-only stages.
- Use a new output directory for each `run`, `fit` or `qc` invocation.
- When reference labels are absent, first check `sample`, the exact mzML spectrum
  IDs and reference peptide notation. Unidentified spectra should remain unlabeled.
- FDR adjustment requires sufficient labeled reference evidence at both levels.
  An error at this step must not be bypassed by substituting a constant factor.
- A checkpoint dimension mismatch indicates an incompatible configuration or
  model; the DDA adapter cannot load a DIA checkpoint as a replacement.

## License

The project retains the GPL-3.0 license of pi-NovoQC. Modified attention layers
are included under their original notices; see `THIRD_PARTY_NOTICES.md`.
Data and model weights are not bundled with the source code.
