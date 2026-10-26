# π-NovoQC: FDR control for de novo peptide sequencing

Code accompanying **[False discovery rate control enables trustworthy deep-learning-based de novo peptide sequencing](https://doi.org/10.64898/2026.06.29.735174)**, by Zhendong Liang, Chengxin Dai, Tianze Ling, Tingpeng Yang, Yun Yang, Yeye Leng, Linhai Xie, Yonghong He, Fuchu He, Yu Wang and Cheng Chang (bioRxiv, 2026).

π-NovoQC implements Counterpart Calibration Theory (CCT) for false discovery rate (FDR) estimation and control at the peptide-spectrum match (PSM) and peptide levels. This repository also provides π-xNovo-DDA and π-xNovo-DIA sequencing-model training code, and a DDA analysis workflow that continues from sequencing and CCT filtering to protein inference and quantification.

**First visit after reading the paper? Start with the [two-level CCT notebook](cct_qc_graphnovo/README.md).** It predicts peptides, constructs target-counterpart features, trains the QC scoring models and exports accepted PSMs and peptides. You do not need to train a sequencing model or perform protein inference to use it.

## Choose your analysis

| What you want to do | Start here | Inputs | Main outputs |
| --- | --- | --- | --- |
| Run PSM/peptide FDR control, including studies that stop at peptides such as SAP analyses | [Standalone notebook](cct_qc_graphnovo/README.md) | DDA MGF spectra, an xNovo-DDA checkpoint and reference PSMs; or an existing paired-feature table | Accepted PSMs and peptides, score tables and fitted QC models |
| Run DDA sequencing, QC, protein inference and quantification | [NovoQC pipeline](novoqc_pipeline/README.md) | DDA mzML, xNovo-DDA checkpoint, reference PSMs, target protein FASTA and database-search peptide intensities | QC results, protein groups and peptide/protein intensities |
| Train or fine-tune the DDA sequencing model | [xNovo-DDA training](xnovo_dda_finetune/README.md) | Annotated DDA MGF chunks; a checkpoint for fine-tuning | Sequencing checkpoint and validation predictions |
| Fine-tune the DIA sequencing model | [xNovo-DIA training](xnovo_dia_finetune/README.md) | DIA training spectra, spectrum-peptide mappings and a pretrained DDA checkpoint | DIA sequencing checkpoint and validation predictions |

Each directory contains its own implementation and can be used independently. NovoQC contains the DDA model code needed for inference. The notebook contains its own inference and CCT code. Neither imports the training directories, and the notebook does not require a NovoQC installation.

## Start with the notebook

Download or clone this repository, then work inside the notebook directory:

```bash
git clone https://github.com/PHOENIXcenter/pi-NovoQC.git
cd pi-NovoQC/cct_qc_graphnovo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The shell examples use Linux/macOS syntax. CUDA is recommended for sequencing large datasets; the notebook also accepts `device: cpu`. For CUDA, install the appropriate PyTorch build for your system before installing the remaining requirements; use a version allowed by `requirements.txt`.

1. Obtain the [GraphNovo data and xNovo-DDA weights](RESOURCES.md). Place the three test-species MGF directories under `data/graphnovo_mgf/` and the checkpoint at `weights/pi_xnovo_dda.pth`, or change these paths in `config.yaml`.
2. Open [config.yaml](cct_qc_graphnovo/config.yaml). Check `checkpoint`, `model_config`, `samples`, `device` and `output`. The default target FDR thresholds are `0.05` for PSMs and `0.05` for peptides. Use a new output directory for each analysis.
3. Start JupyterLab:

```bash
python -m jupyterlab xNovo_DDA_based_CCT_validation.ipynb
```

Run the cells in order. The default output directory is `results/graphnovo_two_level_qc/`. Open `psms.csv` and `peptides.csv` for accepted identifications and `summary.csv` for the analysis summary. Full score tables and fitted QC models are saved alongside them. The [notebook guide](cct_qc_graphnovo/README.md) explains how to use your own spectra or previously generated features.

**Two different models are involved.** The downloaded xNovo-DDA sequencing checkpoint predicts peptide sequences and stays fixed. The notebook trains separate strict and relaxed QC scoring models using reference-labeled PSMs and calculates the FDR adjustment factors. Using the QC notebook does not require sequencing-model training.

## Use your own data

For PSM/peptide QC, follow the [custom-data instructions](cct_qc_graphnovo/README.md#use-your-own-spectra). Provide database-search identifications that have passed the upstream search QC. Spectrum IDs and sample names must match the spectra being analysed. Predictions without database-reference labels remain unlabeled; they are not treated as incorrect predictions.

For protein analysis and quantification, follow the [complete DDA workflow](novoqc_pipeline/README.md#complete-workflow-for-one-dda-sample). Its `novoqc run` command includes all analysis stages, while `novoqc features`, `novoqc fit` and `novoqc qc` allow you to stop at peptides. Protein inference uses a separate target-decoy procedure; the CCT PSM/peptide thresholds do not replace protein-level FDR control. A peptide accepted by QC can be used in SAP analysis, but establishing a SAP requires additional biological analysis.

The included raw-spectrum adapters use xNovo-DDA. A DIA sequencing checkpoint cannot be substituted into these DDA adapters. The pipeline can accept pre-extracted 32-feature DIA tables for QC; the DIA training directory provides sequencing-model training, not a raw-DIA-to-protein workflow.

## How the code relates to the paper

The figure numbers below refer to the accompanying revised manuscript and its Supplementary Information (SI).

| Method in the paper | Paper location | Implementation |
| --- | --- | --- |
| Target-counterpart construction and CCT | Fig. 1; Methods, "π-NovoQC quality-control workflow" | [features.py](novoqc_pipeline/novoqc/features.py), [scoring.py](novoqc_pipeline/novoqc/scoring.py), [qc.py](novoqc_pipeline/novoqc/qc.py); local equivalents in the notebook directory |
| Cross-attention features: 33 for DDA and 32 for DIA | Fig. 3b; Supplementary Table 2 and Supplementary Note 2 | [feature_math.py](novoqc_pipeline/novoqc/feature_math.py), [features.py](novoqc_pipeline/novoqc/features.py) |
| Sequential PSM and peptide QC | Figs. 1d and 3c | [qc.py](novoqc_pipeline/novoqc/qc.py); the [standalone notebook](cct_qc_graphnovo/xNovo_DDA_based_CCT_validation.ipynb) walks through both stages |
| Acquisition-specific π-xNovo models | Fig. 3a and Supplementary Figs. 2-3 | [xnovo_dda_finetune/](xnovo_dda_finetune/README.md), [xnovo_dia_finetune/](xnovo_dia_finetune/README.md) |
| Protein inference and MS1 quantification | Fig. 3c and the DHP application | [proteins.py](novoqc_pipeline/novoqc/proteins.py), [chromatography.py](novoqc_pipeline/novoqc/chromatography.py), [quantification.py](novoqc_pipeline/novoqc/quantification.py) |

Target and counterpart describe the **same predicted peptide and spectrum**. The revised construction extracts both from the same inference cross-attention matrix using the offset row slices `0:L` and `1:1+L`, where `L` is peptide length. It does not require training-mode dropout. Strict and relaxed scoring models determine the combined ranking; the FDR adjustment factor scales the counterpart-to-target count ratio at a score threshold. The counterpart is a feature representation, not a peptide known to be incorrect.

The notebook is a reusable analysis example. Reproducing a particular result in the paper additionally requires the corresponding checkpoint, data split, preprocessing and experimental settings described in the Methods and SI. The default configuration is not a launcher for every experiment or figure in the paper.

## Data, weights and citation

See [RESOURCES.md](RESOURCES.md) for the data repositories, the Google Drive resources, expected checkpoint locations and input formats. The GraphNovo test data and the accompanying CCT resources are provided through the named Google Drive archive; the larger datasets used in the paper remain available from their original repositories.

Please cite the bioRxiv preprint when using this work:

> Liang Z, Dai C, Ling T, Yang T, Yang Y, Leng Y, Xie L, He Y, He F, Wang Y, Chang C. **False discovery rate control enables trustworthy deep-learning-based de novo peptide sequencing.** bioRxiv (2026). DOI: [10.64898/2026.06.29.735174](https://doi.org/10.64898/2026.06.29.735174).

Machine-readable citation information is provided in [CITATION.cff](CITATION.cff).

## Help and licenses

For input requirements, outputs and common setup errors, use the README in the directory you selected above. Report reproducible problems through [GitHub Issues](https://github.com/PHOENIXcenter/pi-NovoQC/issues), including the module, command, error message and relevant configuration. Do not include private data or credentials.

See the root [LICENSE](LICENSE), the module licenses and their third-party notices. Vendored dependencies retain their original license files.
