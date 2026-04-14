#!/usr/bin/env python3
"""
Predict TEC (with uncertainty) and density from composition.

Usage examples:
  python scripts/predict_tec_density.py --comp Fe:64,Ni:29,Co:5,Si:2  
  python scripts/predict_tec_density.py --comp-json '{"Fe":64, "Ni":29, "Co":5, "Si":2}'
  python scripts/predict_tec_density.py --comp-file comps.json

Options:
  --fraction molar|mass   input fraction type (default: molar)
  --models-dir DIR        where models are stored (default: models)
  --iter iter04.GA        pick models from an iteration subdir (default: auto, all tec*.pt under models)
"""
import argparse
import glob
import json
import os
import sys
from typing import Dict, List
import numpy as np
from deepmd.pt.infer.deep_eval import DeepProperty

# Local helpers expected in repo root
from target import comp2struc, pred, get_packing
from constraints_utils import mass_to_molar as mass_to_molar_converter

CONSTANT_DIR = os.path.join(os.path.dirname(__file__), '..', 'constant')
with open(os.path.join(CONSTANT_DIR, 'densities.json'), 'r') as f:
    DENSITIES = json.load(f)


def normalize_composition_dict(comp: Dict[str, float]) -> Dict[str, float]:
    tot = float(sum(comp.values()))
    if tot <= 0:
        raise ValueError('Composition sum must be > 0')
    return {k: float(v)/tot for k, v in comp.items() if float(v) > 0}


def parse_composition_arg(comp_str: str) -> Dict[str, float]:
    comp = {}
    for item in comp_str.split(','):
        if not item.strip():
            continue
        el, val = item.split(':')
        comp[el.strip()] = float(val)
    return comp


def calculate_density_weighted_avg(comp: Dict[str, float], fraction_type: str) -> float:
    # normalize first
    comp = normalize_composition_dict(comp)
    elems = list(comp.keys())
    vals = np.array(list(comp.values()), dtype=float)
    if fraction_type == 'mass':
        vals = np.array(mass_to_molar_converter(vals, elems), dtype=float)
    elif fraction_type != 'molar':
        raise ValueError('fraction_type must be mass or molar')
    return float(sum(vals[i] * DENSITIES[elem] for i, elem in enumerate(elems)))


def load_tec_models(models_root: str, iteration: str = None) -> List[DeepProperty]:
    pattern = 'tec*.pt'
    if iteration:
        search = os.path.join(models_root, iteration, pattern)
    else:
        search = os.path.join(models_root, pattern)
    paths = sorted(glob.glob(search))
    if not paths:
        raise FileNotFoundError(f'No TEC model found at {search}')
    return [DeepProperty(p) for p in paths]


def predict_tec(comp: Dict[str, float], fraction_type: str, models: List[DeepProperty]) -> Dict[str, float]:
    comp = normalize_composition_dict(comp)
    elems = list(comp.keys())
    vals = list(comp.values())
    if fraction_type == 'mass':
        vals = mass_to_molar_converter(np.array(vals, dtype=float), elems).tolist()
    packing = get_packing(elems, vals)
    structs = comp2struc(elems, vals, packing=packing)

    preds = []
    for m in models:
        for s in structs:
            preds.append(float(pred(m, s)))
    return {
        'mean': float(np.mean(preds)),
        'std': float(np.std(preds)),
        'n': len(preds)
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--comp', help='e.g., Fe:64,Ni:29,Co:5,Si:2')
    g.add_argument('--comp-json', help='JSON dict string of composition')
    g.add_argument('--comp-file', help='JSON file path of composition dict')
    ap.add_argument('--fraction', default='molar', choices=['molar', 'mass'])
    ap.add_argument('--models-dir', default=os.path.join(os.path.dirname(__file__), '..', 'models'))
    ap.add_argument('--iter', dest='iteration', default=None, help='e.g., iter04.GA')
    args = ap.parse_args()

    if args.comp:
        comp = parse_composition_arg(args.comp)
    elif args.comp_json:
        comp = json.loads(args.comp_json)
    else:
        with open(args.comp_file, 'r') as f:
            comp = json.load(f)

    tec_models = load_tec_models(args.models_dir, args.iteration)
    tec_stat = predict_tec(comp, args.fraction, tec_models)
    density = calculate_density_weighted_avg(comp, args.fraction)

    out = {
        'input': comp,
        'fraction': args.fraction,
        'iteration': args.iteration,
        'tec': tec_stat,
        'density': density
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
