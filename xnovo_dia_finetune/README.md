# π-xNovo (DIA) fine-tuning code

This is the DIA sequencing component of **False discovery rate control enables
trustworthy deep-learning-based de novo peptide sequencing** (Liang et al.;
Fig. 3a and Supplementary Fig. 3). It fine-tunes the sequencing model; it does
not run the DDA notebook or the raw-DDA NovoQC analysis workflow.

This repository contains the **π-xNovo-DIA** fine-tuning code used in the π-NovoQC project. It also includes optional preprocessing scripts that turn **run-level DIA identifications** (e.g., from DIA-NN) into **spectrum-level training pairs** for model fine-tuning.

## Repository layout

- `main_xNovo.py` — entry point for fine-tuning / training
- `configs/xnovo_dia_finetune.yaml` — training configuration (edit input paths)
- `dataset.py`, `model.py`, `utils.py`, `evaluate.py` — core training code
- `s1_psm_match.py`, `s2_train_data_generation.py`, `s3_train_data_offset.py` — optional data-prep scripts for DIA
- `depthcharge_new/`, `pyteomics_master/` — vendored dependencies (see `THIRD_PARTY_NOTICES.md`)

## Quick start

### 1) Install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use Linux with a CUDA GPU and an appropriate PyTorch build for training. The
launcher uses NCCL; CPU-only and macOS training are not supported by this entry.

### 2) Prepare data (optional helper scripts)

DIA database-search tools often report IDs at the **run level**. For fine-tuning π-xNovo-DIA, we need spectrum-level pairs. The included scripts implement a practical matching strategy using **retention time (RT)** and **isolation-window metadata**.

Recommended folder structure:

```
data/
  db_results_sep/          # per-run database-search outputs (CSV/TSV; edit script to match your format)
  mzml/                    # DIA mzML files
  psm_info/                # generated spectrum–peptide mappings
  train_data/              # generated MGF shards for training
  train_data_offset/       # cached lookup/index structures
```

Example commands (adjust paths/IDs for your dataset):

```bash
# 1) Build spectrum–peptide mappings for a subset of runs
python s1_psm_match.py \
  --db_results_dir data/db_results_sep \
  --mzml_dir data/mzml \
  --out_dir data/psm_info \
  --rt_thresh 1 \
  --id_start 0 --id_end 10

# 2) Generate training MGF shards from mzML + matched PSMs
python s2_train_data_generation.py \
  --mzml_dir data/mzml \
  --psm_info_dir data/psm_info \
  --train_data_dir data/train_data \
  --id_start 0 --id_end 10

# 3) Build offset/index lookup tables (optional speed-up)
python s3_train_data_offset.py \
  --train_data_dir data/train_data \
  --psm_info_dir data/psm_info \
  --out_dir data/train_data_offset \
  --id_start 0 --id_end 10
```

### 3) Fine-tune the model

Edit `configs/xnovo_dia_finetune.yaml` so that:

- `mgf_file` points to your generated `data/train_data/`
- `psm_path` points to your `data/psm_info/`
- `lookup_file` points to your `data/train_data_offset/`

Obtain a compatible DDA initialization checkpoint from the
[author-provided weights folder](https://drive.google.com/drive/folders/1pfrtLEEoWrGrIfriNGkIJqAIRgrH2tDW?usp=drive_link).
For `--pretrain_model MassIVE_KB`, place it at
`pretrained_DDA_weights/MassIVE_KB_trained_xNovo_DDA.pth`. The alternative
`--pretrain_model TimsTOF_DDA` expects
`pretrained_DDA_weights/TimsTOF_DDA_pretrained.pth`. Use the checkpoint matching
the selected option and architecture; the filenames are local loader paths.

Choose a validation sample that exists in your generated training inputs. Pass
that name with `--valid_species`, which overrides YAML `valid_species`.

Then run:

```bash
torchrun --standalone --nproc_per_node=1 main_xNovo.py \
  --multiprocessing True --n_gpus 1 \
  --config configs/xnovo_dia_finetune.yaml \
  --pretrain_model MassIVE_KB --valid_species Hh_24h_500_04 \
  --max_epochs 10 --bs 32 --n_peaks 600 --sub_peaks 150
```

Replace `Hh_24h_500_04` with your validation sample. CLI options also override
the YAML values for peak counts, layer count, dropout, sampling, batch size,
workers, learning rate, epoch count and masking; check `python main_xNovo.py
--help` and set the relevant options explicitly. The command above is a usage
example; match the Methods/SI settings when reproducing a paper experiment.

For initial fine-tuning, leave YAML `modelname: false`. To resume a DIA training
checkpoint, set `modelname` to that file: the loader restores its model state,
epoch and optimizer state. The DDA initialization file is still read by the
current loader before the DIA state is restored.

The preserved DIA training loop uses a distributed sampler and a wrapped model,
including when training on one GPU. The command above starts one CUDA worker
and supplies that environment explicitly. A plain `python main_xNovo.py` command
does not supply the required distributed environment. Use the one-GPU command
above for this workflow.

This directory contains its own model, training code and vendored dependencies.
It does not import either NovoQC or the notebook example, and does not perform
PSM/peptide FDR control or quantification.

## Notes

- `checkpoints/` contains the trained sequencing checkpoint; `test_results/`
  contains validation predictions, `log/` contains TensorBoard logs, and
  `config_file/` contains the effective configuration snapshots. Checkpoint
  selection follows training loss in this entry point.
- A resulting DIA checkpoint is not compatible with the raw-DDA feature
  adapters in the standalone notebook or NovoQC pipeline. DIA QC needs the
  corresponding DIA feature extraction; the pipeline accepts already extracted
  32-feature tables through `fit --acquisition dia`.

- The scripts are **research code** released for reproducibility. For new datasets, you may need to adapt column names / formats in the data-prep scripts.
- Output directories such as `checkpoints/`, `test_results/`, and `config_file/` are ignored by git.

## Citation

See `CITATION.cff`.

## License

See `LICENSE`. Vendored third-party packages keep their original licenses
(see `THIRD_PARTY_NOTICES.md` and license files in the vendored directories).
