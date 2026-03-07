import dpdata
import os
import glob

ms_val = dpdata.MultiSystems().load_systems_from_file('test', fmt='deepmd/npy/mixed')
for idx, ss in enumerate(ms_val):
    ss.to_deepmd_npy(f'test_npy/{idx}')

new = glob.glob('new_data*/*')
for nn in new:
    if "_npy" in nn:
        continue
    base = nn.split('/')[0]
    left = nn.split('/')[-1]
    try:
        ms_new = dpdata.MultiSystems().load_systems_from_file(nn, fmt='deepmd/npy/mixed')
        for idx, ss in enumerate(ms_new):
            print(ss)
            ss.to_deepmd_npy(f'{base}_npy/{left}')
    except Exception as e:
        print(nn, e)
