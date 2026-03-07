import dpdata
import random
import numpy as np
import glob
from tqdm import tqdm
from sklearn.model_selection import KFold

# Constants
NUM_FOLDS = 5
SEED = 42
random.seed(SEED)

ms_tst = dpdata.MultiSystems()
ms_trn_folds = [dpdata.MultiSystems() for _ in range(NUM_FOLDS)]
datasets = glob.glob('datasets/*')

# Pre-define test dataset
tst_ratio = 0.02
num_samples = len(datasets)
num_tst_samples = int(tst_ratio * num_samples)

# Shuffle datasets and separate test set
random.shuffle(datasets)
tst_datasets = datasets[:num_tst_samples]
remaining_datasets = datasets[num_tst_samples:]

# Load test datasets
for dataset in tqdm(tst_datasets, desc="Loading test datasets"):
    ms_temp = dpdata.MultiSystems().load_systems_from_file(dataset, fmt='deepmd/npy/mixed')
    ms_tst.append(ms_temp)

# Perform 5-fold splitting on the remaining datasets
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
fold_indices = list(kf.split(remaining_datasets))

# Load and distribute datasets into folds
for fold, (train_idx, val_idx) in enumerate(fold_indices):
    # for idx in train_idx:
    #     dataset = remaining_datasets[idx]
    #     ms_temp = dpdata.MultiSystems().load_systems_from_file(dataset, fmt='deepmd/npy/mixed')
    #     ms_trn_folds[fold].append(ms_temp)

    for idx in val_idx:
        dataset = remaining_datasets[idx]
        ms_temp = dpdata.MultiSystems().load_systems_from_file(dataset, fmt='deepmd/npy/mixed')
        ms_trn_folds[fold].append(ms_temp)

# Save datasets for each fold
for fold, ms_fold in enumerate(ms_trn_folds):
    fold_name = f"fold_{fold + 1}"
    ms_fold.to_deepmd_npy_mixed(f"train_{fold_name}")
    print(f"Saved fold {fold_name} with {len(ms_fold)} systems.")

# Save the test dataset
ms_tst.to_deepmd_npy_mixed('test')
print(f"Test dataset saved with {len(ms_tst)} systems.")
