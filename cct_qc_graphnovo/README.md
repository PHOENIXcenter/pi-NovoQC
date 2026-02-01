# CCT QC demo on the GraphNovo test set

This repository provides a **minimal, runnable example** of **Cognitive Competition Theory (CCT)**–based quality control (QC) for **generative de novo peptide sequencing**.

The example is implemented as a single notebook:

- **`xNovo_DDA_based_CCT_validation.ipynb`** — GraphNovo test-set validation for CCT QC (PSM-level + peptide-level).

## Repository layout

- `xNovo_DDA_based_CCT_validation.ipynb` — main demo notebook  
- `config_xNovo_DDA.yaml` — configuration template  
- `model.py` — π-xNovo-DDA model and feature extraction  
- `FDR.py` — CCT scoring, FDR estimation, and interval adjustment  
- `evaluate.py` — evaluation helpers (AA matching / peptide recall)  
- `BLAST.py` — optional peptide-to-protein mapping helpers  
- `depthcharge_new/` and `pyteomics_master/` — vendored dependencies (see `THIRD_PARTY_NOTICES.md`)  

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

> The code was developed for GPU runs, but the GraphNovo demo can be executed on CPU for small-scale tests.

### 2) Prepare data

The GraphNovo benchmark data are **not** included in this repository. Please download them from the original GraphNovo release and prepare the MGF files on your machine.

Set `mgf_file` in `config_xNovo_DDA.yaml` to point to your local MGF root directory.

Example folder structure:

```
data/graphnovo_mgf/
  Human_HeLa/
    *.mgf
  Mouse_Cerebellum/
    *.mgf
  A_Thaliana/
    *.mgf
  C_Elegans/
    *.mgf
  E_Coli/
    *.mgf
```

The notebook uses `valid_species` to define the held-out test subset.

### 3) Run the notebook

```bash
jupyter lab
```

Open `xNovo_DDA_based_CCT_validation.ipynb` and run the cells top-to-bottom.

Outputs will be written to local folders such as `checkpoints/`, `test_results/`, and `config_file/` (ignored by git).

## Optional: BLAST

Some downstream mapping steps use BLAST+. If you run them, install **NCBI BLAST+** and make sure `makeblastdb` / `blastp` are available. The helper functions default to calling BLAST via `conda run -n blast ...`; edit `conda_env` if you do not use conda.

## Citation

If you use this code, please cite the π-NovoQC / CCT manuscript (see `CITATION.cff`).

## License

See `LICENSE`. Third-party packages included in this repository keep their original licenses (see `THIRD_PARTY_NOTICES.md`).
