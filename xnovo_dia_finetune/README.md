# π-xNovo (DIA) fine-tuning code

This repository contains the **π-xNovo-DIA** fine-tuning code used in the π-NovoQC project. It also includes optional preprocessing scripts that turn **run-level DIA identifications** (e.g., from DIA-NN) into **spectrum-level training pairs** for model fine-tuning.

## Repository layout

- `main_xNovo.py` — entry point for fine-tuning / training
- `config_xNovo.yaml` — training configuration (edit paths for your machine)
- `dataset.py`, `model.py`, `utils.py`, `evaluate.py` — core training code
- `s1_psm_match.py`, `s2_train_data_generation.py`, `s3_train_data_offset.py` — optional data-prep scripts for DIA
- `depthcharge_new/`, `pyteomics_master/` — vendored dependencies (see `THIRD_PARTY_NOTICES.md`)

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

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

Edit `config_xNovo.yaml` so that:

- `mgf_file` points to your generated `data/train_data/`
- `psm_path` points to your `data/psm_info/`
- `lookup_file` points to your `data/train_data_offset/`

Then run:

```bash
python main_xNovo.py --config config_xNovo.yaml
```

#### Multi-GPU

`torch.distributed.launch` has been deprecated upstream. Prefer `torchrun` when possible:

```bash
torchrun --nproc_per_node 2 main_xNovo.py --multiprocessing True --n_gpus 2 --config config_xNovo.yaml
```

## Notes

- The scripts are **research code** released for reproducibility. For new datasets, you may need to adapt column names / formats in the data-prep scripts.
- Output directories such as `checkpoints/`, `test_results/`, and `config_file/` are ignored by git.

## Citation

See `CITATION.cff`.

## License

See `LICENSE`. Vendored third-party packages keep their original licenses (see `THIRD_PARTY_NOTICES.md`).
