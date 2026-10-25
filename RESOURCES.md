# Data and model resources

## First notebook run

The default notebook analyses the GraphNovo test species *Arabidopsis thaliana*, *Caenorhabditis elegans* and *Escherichia coli*.

| Resource | Source | Local destination |
| --- | --- | --- |
| GraphNovo test data and CCT resources | [src_data_and_weights_for_cct_qc_graphnovo.zip](https://drive.google.com/file/d/1zUlTf8gUY6ntlKspGJyfpySJ6n1DTJ-h/view) | Extract the test MGF files and the accompanying checkpoint, then set the paths in `config.yaml` |
| DDA pretrained models | [pretrained_model_for_DDA.zip](https://drive.google.com/file/d/1YLYgdNoA5B12j7xW-HM-2oapZlQsHZ8L/view) | Extract the DDA checkpoint required by the training or inference entry point |

The complete shared folder is [here](https://drive.google.com/drive/folders/1pfrtLEEoWrGrIfriNGkIJqAIRgrH2tDW?usp=drive_link). It contains two named archives:

- [src_data_and_weights_for_cct_qc_graphnovo.zip](https://drive.google.com/file/d/1zUlTf8gUY6ntlKspGJyfpySJ6n1DTJ-h/view): the GraphNovo test data and CCT example resources used by the standalone notebook.
- [pretrained_model_for_DDA.zip](https://drive.google.com/file/d/1YLYgdNoA5B12j7xW-HM-2oapZlQsHZ8L/view): pretrained DDA sequencing models used for DDA inference and training initialization.

Download and extract the archive required by the workflow you selected. Use the
code and notebook from this repository, and set the corresponding data and
checkpoint paths in the local configuration.

Set `checkpoint` to the extracted DDA checkpoint that matches the model
configuration. The checkpoint and its configuration must use the same acquisition
type, architecture and modification vocabulary.

```text
cct_qc_graphnovo/
  config.yaml
  weights/
    pi_xnovo_dda.pth
  data/graphnovo_mgf/
    A_Thaliana/
    C_Elegans/
    E_Coli/
```

The notebook accepts MGF files in each configured species directory; it does not require the chunk naming used by the training loaders. Preserve the MGF `TITLE`, `PEPMASS`, `CHARGE` and peak data. For the annotated benchmark, `SEQ` provides the reference peptide used to label the model prediction. Reference peptides are not passed to the sequencing model as predictions.

## Checkpoint selection by module

| Module | Configuration or option | Required checkpoint |
| --- | --- | --- |
| Standalone notebook | `checkpoint` and `model_config` in `config.yaml` | xNovo-DDA sequencing checkpoint matching the architecture and residue vocabulary |
| NovoQC DDA pipeline | `--checkpoint`; optional `--model-config` | xNovo-DDA sequencing checkpoint matching the supplied model configuration |
| DDA fine-tuning | YAML `modelname`, or `--pretrain_model MassIVE_KB` / `TimsTOF_DDA` | Matching DDA checkpoint; the two named options expect files under `weights/` as documented in the training README |
| DIA fine-tuning | `--pretrain_model MassIVE_KB` / `TimsTOF_DDA` | DDA initialization checkpoint under `pretrained_DDA_weights/`; see the DIA README for exact filenames and resume behavior |

Sequencing checkpoints and QC scoring models are different files. The notebook and pipeline generate the QC models and FDR adjustment factors locally. Do not pass a QC model as `--checkpoint`. Architecture dimensions and modification tokens must agree with sequencing-model training. Only load checkpoints from a trusted source.

## Other datasets used in the manuscript

These are the original data sources cited in the manuscript. The Methods and
Supplementary Note 3 specify the subsets and preprocessing used for each analysis.

| Dataset | Repository or accession |
| --- | --- |
| GraphNovo full dataset | [Zenodo 8000316](https://doi.org/10.5281/zenodo.8000316) |
| GraphNovo test data used by the notebook | [Google Drive archive](https://drive.google.com/file/d/1zUlTf8gUY6ntlKspGJyfpySJ6n1DTJ-h/view) |
| MassIVE-KB | [MassIVE Knowledge Base spectral library v1](https://massive.ucsd.edu/ProteoSAFe/static/massive-kb-libraries.jsp), "All Candidate library spectra" |
| Deep Human Proteome (DHP) | ProteomeXchange PXD024364 |
| Astral DDA/DIA | ProteomeXchange PXD046453 |
| Dong-Ecoli-QE | ProteomeXchange PXD008782 |
| Orbitrap DIA | CPTAC/PDC PDC000200 |
| timsTOF DDA | iProX IPX0018169000; ProteomeXchange PXD080632 |
| Nine-species dataset | PRIDE PXD005025, PXD004948, PXD004325, PXD004565, PXD004536, PXD004947, PXD003868, PXD004467 and PXD004424 |

For your own data, follow the [notebook input guide](cct_qc_graphnovo/README.md#use-your-own-spectra) or the [pipeline input schemas](novoqc_pipeline/README.md#complete-workflow-for-one-dda-sample). Database search and its upstream FDR filtering precede reference-PSM input to CCT.
