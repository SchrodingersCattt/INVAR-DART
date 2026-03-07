# Finetune (TEC/property) for INVAR

This folder packages the **finetune pipeline** (data prep → DeepMD-PyTorch training → inference/evaluation) that was used in `iter04.finetune/dpa_v3_1`, cleaned for public release.

## Layout

- `scripts/`: data prep, training utilities, inference/evaluation
- `config/input.json`: DeepMD (PyTorch backend) training configuration
- `models/pretrained/`: pretrained checkpoint(s) used for finetune
- `models/finetuned/`: finetuned checkpoint(s) (e.g., 5-fold)
- `data/raw/`: put the raw Excel here (`Data_base_DFT_Thermal.xlsx`)
- `data/processed/`: generated intermediate outputs (json/npy); **not meant to be committed**
- `runs/`: training outputs (`workspace*/`, logs); **not meant to be committed**

## Requirements

Python packages (typical):
- `deepmd-kit` (PyTorch backend)
- `dpdata`
- `pymatgen`
- `numpy`, `pandas`, `matplotlib`, `tqdm`

## Quick start (local)

Run from this directory (`public_release/INVAR-DART/finetune`).

### 1) Data preparation

1. Place the Excel file at:
   - `data/raw/Data_base_DFT_Thermal.xlsx`

2. Generate structures / mixed systems (writes to `data/processed/` and/or `new_data*/` depending on script options):
   - `python scripts/00.mk_new_data.py`

3. Split folds:
   - `python scripts/01.split_5-fold.py`

4. Convert to DeepMD NPY / property NPY:
   - `python scripts/02.mixed_to_npy.py`
   - `python scripts/02.prep_prop_npy.py`

5. Assemble final DeepMD dataset sets:
   - `python scripts/03.mk_set.py`

> The scripts will be further parameterized (paths, seeds, output dirs). For now they are the direct reference copies.

### 2) Training (finetune)

Example (single run):

- `dp --pt train config/input.json --finetune models/pretrained/DPA-3.1-3M.pt`

Typical 5-fold idea (bash pseudo-example):

- for each fold $k$, set fold-specific training/validation system lists (see `config/input.json` and data split outputs), then run:
  - `dp --pt train config/input.json --finetune models/pretrained/DPA-3.1-3M.pt`

### 3) Inference / evaluation

- `python scripts/infer_by_models.py`

This script searches `workspace*/model.ckpt.pt` and evaluates on `test_npy/*`.

## Cluster command explanation (no VOC specifics)

You can run 5 folds in parallel by launching 5 independent processes, each in its own working directory:

- `cd runs/fold_0 && dp --pt train ../../config/input.json --finetune ../../models/pretrained/DPA-3.1-3M.pt`
- `cd runs/fold_1 && dp --pt train ../../config/input.json --finetune ../../models/pretrained/DPA-3.1-3M.pt`
- ...

If you use a scheduler (Slurm/PBS/etc.), wrap each fold command into a job script and request GPU/CPU resources as appropriate.

## Notes

- Large artifacts (training workspaces, logs, processed data) should be ignored by git.
- If you want to publish large model checkpoints, consider Git LFS.
