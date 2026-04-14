# INVAR-DART

This release contains code for composition optimization (GA/NSGA/PSO/BO) and the accompanying property prediction tooling.

## Modules

- Optimization (GA/NSGA/PSO/BO): see the scripts in this directory (e.g., `ga.py`, `nsga.py`, `pso.py`, `bo_optuna.py`).
- Finetune (property/TEC model): see `finetune/README.md`.

## Predict TEC and density from composition

We provide pretrained TEC models organized by iteration under `models/iterXX.GA/tec_*.pt` and a simple CLI tool `scripts/predict_tec_density.py`.

Examples:

- Predict with default models directory (all `models/tec*.pt`):
	- `python scripts/predict_tec_density.py --comp Fe:64,Ni:29,Co:5,Si:2 --fraction molar`

- Predict using a specific iteration model set (e.g., iter04.GA):
	- `python scripts/predict_tec_density.py --comp Fe:64,Ni:29,Co:5,Si:2 --fraction molar --iter iter04.GA`

Input format alternatives:

- JSON string: `--comp-json '{"Fe":64, "Ni":29, "Co":5, "Si":2}'`
- JSON file: `--comp-file comps.json`

Output is a JSON containing mean/std over model and structural sampling, and the weighted-average density. Ensure `constant/densities.json` and helper modules (`target.py`, `constraints_utils.py`) are available in the repository root so the script can import them.
