# π-xNovo-DDA: sequencing-model training and fine-tuning

This repository contains the **DDA** implementation of the π-xNovo de novo peptide sequencing model, together with a minimal training/evaluation script.

This is the DDA sequencing component of **False discovery rate control enables
trustworthy deep-learning-based de novo peptide sequencing** (Liang et al.;
Fig. 3a and Supplementary Fig. 2). It trains a sequencing model, not the CCT
scoring models. To analyse spectra with an existing sequencing checkpoint,
use the independent CCT notebook or NovoQC pipeline instead.

## What is in this repo?

- `main_xNovo.py` — entry point for fine-tuning (kept intentionally simple).
- `xnovo_dda/` — model, dataset loader, training loops, and evaluation utilities.
- `configs/` — example YAML config.

## Installation

### 1) Create an environment

This code was developed for **GPU (CUDA) training**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The required `depthcharge_new` fork is included in this directory. It does not
depend on either NovoQC or the notebook example. Install dependencies from this
directory and run the training command here so that local imports resolve.

### 2) Check GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

## Data format

The loader expects MGF files stored as **chunks** under a root directory:

```
data/mgf_dda/
  SAMPLE_A_DDA_.../
    0_1000.mgf
    1000_2000.mgf
    ...
  SAMPLE_B_DDA_.../
    0_1000.mgf
    ...
```

Each MGF entry should include:
- `m/z array`
- `intensity array`
- `params.pepmass` (precursor m/z)
- `params.charge`
- `params.seq` (peptide string; modified residues allowed)

These are parsed MGF fields: the file contains `PEPMASS`, `CHARGE`, `SEQ` and
peak m/z-intensity rows inside each `BEGIN IONS` / `END IONS` block. Chunk names
use an exclusive upper endpoint, and `spec_step` defaults to 1000. A sample with
2,500 spectra therefore contains `0_1000.mgf`, `1000_2000.mgf` and
`2000_2500.mgf`. The loader infers the total from these names; arbitrary MGF
filenames or inclusive endpoints do not satisfy this training-input format.

## Fine-tuning

1) Edit `configs/xnovo_dda_finetune.yaml`:
- `mgf_file`: path to your MGF root directory
- `valid_species`: the *subfolder name* to use as a validation set
- `modelname`: checkpoint path for fine-tuning, or `false` for training from scratch

The [author-provided weights folder](https://drive.google.com/drive/folders/1pfrtLEEoWrGrIfriNGkIJqAIRgrH2tDW?usp=drive_link)
is linked from the project repository. Select a compatible DDA checkpoint and
set its local path in `modelname`. Match the architecture and residue vocabulary
to that checkpoint. The default YAML has `modelname: false` and therefore starts
from random initialization; it is not a pretrained fine-tuning run until a
checkpoint is supplied.

2) Run fine-tuning:

```bash
python main_xNovo.py \
  --config configs/xnovo_dda_finetune.yaml \
  --device cuda:0 \
  --max_epoch 10 \
  --dropout 0.1
```

The example explicitly sets training dropout and the epoch count; these are
usage examples, not a claim to reproduce every paper experiment. CLI options
`--max_epoch`, `--dropout` and `--train_mask_ratio` override the corresponding
YAML values even when the CLI defaults are used. Set them explicitly for your
experiment. `valid_species` is read from the YAML.

Outputs:
- `checkpoints/` — best checkpoint (by training loss)
- `log/` — TensorBoard logs
- `test_results/` — CSV of validation predictions
- `config_file/` — dumped config snapshots (for provenance)

## Training scope

The documented entry point trains on one device. Set `--device cuda:0` for a
single CUDA GPU. The historical distributed branches are retained as source;
they are not the supported launcher for this package. Checkpoints selected by
`--pretrain_model` are read from `weights/`, with no author-specific paths.

`--pretrain_model MassIVE_KB` loads
`weights/MassIVE_KB_trained_xNovo_DDA.pth`; `--pretrain_model TimsTOF_DDA` loads
`weights/TimsTOF_DDA_pretrained.pth`. These options override YAML `modelname`.
To use your own checkpoint path, keep the default `--pretrain_model False`.

The saved sequencing checkpoint can be used with the independent notebook or
NovoQC pipeline together with a matching model configuration. QC scoring models
and FDR adjustment factors are fitted there using reference PSMs.

The sequencing model is independent of QC. This directory does not calculate
FDR adjustment factors, filter PSMs/peptides or perform quantification.

## Citation

Please cite Liang et al., **False discovery rate control enables trustworthy
deep-learning-based de novo peptide sequencing**. See `CITATION.cff`.

## License

See the preserved `LICENSE` and licenses under `depthcharge_new` and
`pyteomics_master`. The bundled Pyteomics compatibility changes use the public
`pandas.__version__` attribute and `pkgutil.extend_path` for its namespace.
