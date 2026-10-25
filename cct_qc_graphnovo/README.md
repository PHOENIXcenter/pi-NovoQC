# Standalone PSM and peptide CCT quality control

This example accompanies **[False discovery rate control for trustworthy AI-based
de novo peptide sequencing](https://doi.org/10.64898/2026.06.29.735174)** (Liang et al., bioRxiv, 2026). It implements
the paired features and sequential PSM/peptide QC described in Figs. 1 and 3,
Supplementary Table 2 and Supplementary Note 2. See `CITATION.cff` for citation
information. The analysis ends at peptides; protein inference is not required.

This directory is independently usable. Copy this entire directory, including
`cct_qc`, to another location; no xNovo training directory or NovoQC installation
is required. The local `cct_qc` package contains xNovo-DDA inference, paired
feature extraction, strict/relaxed scoring and two-level CCT filtering.

## Installation

Install an appropriate PyTorch build for your hardware, then run:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab xNovo_DDA_based_CCT_validation.ipynb
```

Open `xNovo_DDA_based_CCT_validation.ipynb` from this directory. Edit
`config.yaml` and run the cells from top to bottom. Relative paths are resolved
from this directory. Use a new output directory for each run.

## Inputs

Download the GraphNovo test data and CCT resources from
[`src_data_and_weights_for_cct_qc_graphnovo.zip`](https://drive.google.com/file/d/1zUlTf8gUY6ntlKspGJyfpySJ6n1DTJ-h/view).
The DDA model archive is
[`pretrained_model_for_DDA.zip`](https://drive.google.com/file/d/1YLYgdNoA5B12j7xW-HM-2oapZlQsHZ8L/view).
Extract the checkpoint that matches `cct_qc/model_config.yaml`, then set its
path in `config.yaml`. The full GraphNovo dataset is available from
[Zenodo 8000316](https://doi.org/10.5281/zenodo.8000316). Full-dataset inference
benefits from a CUDA GPU; set `device: cuda:0` after installing a compatible
PyTorch build.

The default configuration generates paired features from GraphNovo MGF spectra.
Place each species' MGF files under `data/graphnovo_mgf/<species>/` and a trusted
xNovo-DDA checkpoint at `weights/pi_xnovo_dda.pth`. The three species are configured
explicitly in `config.yaml`. The MGF `SEQ` field supplies the reference peptide
for this annotated benchmark; it is not the de novo prediction.

## Use your own spectra

For unannotated MGF spectra, set `use_mgf_reference: false` and provide qualified
reference PSMs separately through `reference_csv`, or include separate annotated
reference spectra in the configured inputs. Reference CSV columns are `sample`,
`spectrum_id` and `reference_sequence`. Spectrum IDs must match MGF `TITLE` values.
Unannotated predictions retain missing labels and are not treated as incorrect.

For example, replace `samples` in `config.yaml` with your sample name and MGF
directory, and set the reference CSV path:

```yaml
use_mgf_reference: false
reference_csv: data/reference_psms.csv
samples:
  - sample: sample_A
    mgf_dir: data/my_mgf/sample_A
```

Use these CSV columns (the row below illustrates the format only):

```csv
sample,spectrum_id,reference_sequence
sample_A,scan_1001,PEPTIDER
```

`scan_1001` must exactly match the corresponding MGF `TITLE`; keep one qualified
reference peptide per spectrum. Include only database identifications that
passed the upstream database-search QC. There must be reference-labeled pairs
available for fitting, even when most spectra have no database identification.

## Use existing paired features

Alternatively, set `generate_features: false` and supply `feature_csv`, containing
one `target` and one `counterpart` row per `(sample, psm_id)`. Both rows describe
the same de novo sequence and share its database-reference correctness label,
if available. The 33 feature columns are listed in `cct_qc/features.py`.

## Feature construction

For each predicted peptide and spectrum, both paired representations come from
the same inference cross-attention matrix. For peptide length `L`, target uses
`0:L` and counterpart uses `1:1+L`. No training-mode dropout is used to generate
the counterpart. The notebook does not import or call a sibling project.

## Notebook stages and outputs

1. Generate or read paired features.
2. Train strict and relaxed PSM scoring models and estimate the PSM FDR
   adjustment factor from reference-labeled pairs.
3. Filter PSMs, recover both members of each selected pair, construct peptide
   representatives and train the peptide scoring models.
4. Estimate the peptide FDR adjustment factor and filter peptides.
5. Export accepted PSMs and peptides, full score tables and fitted QC models.

The notebook uses the same stage functions and feature definitions as NovoQC,
stored locally here. Target FDR thresholds default to 5% at each level. Set them
in `config.yaml`; QC training uses seed 454. Reference labels are needed to train
the scoring models and calculate the adjustment factors. The stored scoring
models and adjustment factors are then applied to all configured candidates.
If the supplied reference set does not yield a finite positive adjustment
factor, the notebook stops with an error instead of substituting a value.

`psms.csv` and `peptides.csv` are the analysis endpoints. No protein inference,
protein FDR filter or quantification is performed. Peptide-level results can be
used in SAP analysis or other studies that do not require protein aggregation.

| Output under `output` | What to use it for |
| --- | --- |
| `paired_features.csv` | Generated feature pairs; present when `generate_features: true` |
| `psms.csv` | Target PSMs accepted at the PSM target FDR threshold |
| `peptides.csv` | Target peptides accepted after the second QC stage |
| `psm_scores.csv`, `peptide_scores.csv` | Scored target/counterpart entries and count-based FDR estimates |
| `summary.csv` | Counts, adjustment factors and observed FDP where reference labels are available |
| `models/` | Strict/relaxed scoring models for both levels and their saved settings |

An FDR estimate is used for filtering; observed FDP is calculated only where
correctness labels exist. A missing observed FDP for unlabeled predictions is
not a measured zero error rate.

## Common setup issues

- If `config.yaml` or `cct_qc` is not found, launch JupyterLab from this directory
  and use the Python environment in which its requirements were installed.
- If the output directory exists, choose a new `output` path before rerunning.
- If a checkpoint fails to load, check the acquisition type, architecture and
  modification vocabulary against `model_config`; do not substitute a DIA model.
- If FDR adjustment cannot be estimated, check spectrum-ID matching and the
  availability of labeled correct/incorrect predictions at both QC stages.
  Reducing the input to a few spectra may leave insufficient reference evidence.

The shipped example illustrates running the method. Reproducing a paper result
requires the checkpoint, data split and settings specified for that experiment.
