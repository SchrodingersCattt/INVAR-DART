import glob
import os
import json
import shutil
#orig_list = glob.glob('valid/*')

def formatting(dataset):
    list_for_training = []
    if isinstance(dataset, str):
        for root, dirs, files in os.walk(dataset):
            for dir in dirs:
                if 'set.' in dir:
                    list_for_training.append(os.path.abspath(root))

    elif isinstance(dataset, list):
        for dd in dataset:
            for root, dirs, files in os.walk(dd):
                for dir in dirs:
                    if 'set.' in dir:
                        list_for_training.append(os.path.abspath(root))
    return list(set(list_for_training))

def rename_datasets(datasets):
    for dataset in datasets:
        for r, d, f in os.walk(dataset):
            for file in f:
                if "property.npy" in f:
                    shutil.copy(os.path.join(r, file), os.path.join(r, f"{PROP_NAME}.npy"))
FOLDs = 5
PROP_NAME = "tec"


for i in range(FOLDs):
    os.makedirs('workspace{}'.format(i), exist_ok=True)
    shutil.copy('input.json', 'workspace{}/input.json'.format(i))

inputs = glob.glob('workspace*/input.json')
dataset_candidates = glob.glob('train_fold_*')
new_datasets = [nn for nn in glob.glob("new_data*/*") if not "_npy" in nn]

for idx, input in enumerate(inputs):

    val = dataset_candidates[idx]
    trn = list(set(dataset_candidates) - set([val]))

    with open(input, 'r') as ff:
        data = json.load(ff)

    data['training']['training_data']['systems'] = formatting(trn) + formatting(new_datasets)
    data['training']['validation_data']['systems'] = formatting(val)

    with open(input, 'w') as ff:
        json.dump(data, ff, indent=4)