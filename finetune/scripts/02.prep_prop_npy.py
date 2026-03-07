import shutil
import glob
from tqdm import tqdm
import os

#datasets_path = glob.glob(f'{datasets_path}/*/*/set*/')
datasets_path = \
    glob.glob(f'train*/*/*/') + glob.glob(f'valid*/*/*/') + glob.glob(f'test*/*/*/') +\
    glob.glob("new_data*/*/*/*/") + glob.glob("new_data*npy/*/*/")
print(datasets_path)
for dd in tqdm(datasets_path):
    fake_energy_file = dd + 'energy.npy'
    property_file = dd + 'property.npy'   
    if not os.path.exists(property_file):
        shutil.copyfile(fake_energy_file, property_file)
    
print('Done')
