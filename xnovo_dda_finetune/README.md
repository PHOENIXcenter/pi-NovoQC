# π-xNovo (DDA) — fine-tuning code

This repository contains the **DDA** implementation of the π-xNovo de novo peptide sequencing model, together with a minimal training/evaluation script.

> **One-command fine-tuning:** `python main_xNovo.py --config configs/xnovo_dda_finetune.yaml`

## What is in this repo?

- `main_xNovo.py` — entry point for fine-tuning (kept intentionally simple).
- `xnovo_dda/` — model, dataset loader, training loops, and evaluation utilities.
- `configs/` — example YAML config.

## Installation

### 1) Create an environment

This code was developed for **GPU (CUDA) training**.

```bash
pip install -r requirements.txt
```

> ⚠️ This code imports a package called `depthcharge_new` (a local fork). For public release, either vendor the fork into this repository or refactor imports to an installable dependency.

### 2) Check GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

## Data format

The loader expects MGF files stored as **chunks** under a root directory:

```
data/mgf_dda/
  SAMPLE_A_DDA_.../
    0_999.mgf
    1000_1999.mgf
    ...
  SAMPLE_B_DDA_.../
    0_999.mgf
    ...
```

Each MGF entry should include:
- `m/z array`
- `intensity array`
- `params.pepmass` (precursor m/z)
- `params.charge`
- `params.seq` (peptide string; modified residues allowed)

## Fine-tuning

1) Edit `configs/xnovo_dda_finetune.yaml`:
- `mgf_file`: path to your MGF root directory
- `valid_species`: the *subfolder name* to use as a validation set

2) Run fine-tuning:

```bash
python main_xNovo.py \
  --config configs/xnovo_dda_finetune.yaml \
  --device cuda:0 \
  --max_epoch 10 \
  --dropout 0.1
```

Outputs:
- `checkpoints/` — best checkpoint (by training loss)
- `log/` — TensorBoard logs
- `test_results/` — CSV of validation predictions
- `config_file/` — dumped config snapshots (for provenance)

## Multi-GPU training (optional)

The original internal version used `torch.distributed.launch` (deprecated). If you need multi-GPU, prefer `torchrun`:

```bash
torchrun --nproc_per_node=8 main_xNovo.py --config configs/xnovo_dda_finetune.yaml
```

## Citation

If you use this code, please cite the corresponding paper (π-NovoQC / CCT).

## License

Add a license before public release (e.g., MIT or Apache-2.0), depending on your institution's policy.
