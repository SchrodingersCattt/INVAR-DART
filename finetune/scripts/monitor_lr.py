import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import glob


lcurves = glob.glob('w*/lcurve.out')

def smooth(x, window_len=100):
    try:
        return savgol_filter(x, window_length=window_len, polyorder=1)
    except:
        return x


plt.figure(figsize=(8, 9))
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Arial'
#plt.style.use('seaborn-whitegrid')

for idx, lcurve in enumerate(lcurves):
    param = lcurve.split('/')[-2]
    print(param)
    lcurve = np.loadtxt(lcurve)
    df = pd.DataFrame(lcurve)
    df.columns = ['step', 'mae_val', 'mae_trn', 'mape_val', 'mape_trn',  'rmse_val', 'rmse_trn', 'lr']
    # Applying the smoothing function to the RMSE values
    smoothed_rmse_e_trn = smooth(df['mape_trn'])
    smoothed_rmse_e_val = smooth(df['mape_val'])
    plt.subplot(3, 3, idx+1)
    plt.plot(df['step'], smoothed_rmse_e_trn, c='#4455fb', linestyle='-', label=r'trn')
    plt.plot(df['step'], smoothed_rmse_e_val, c='#4455fb', linestyle='-', label=r'val', alpha=0.6)

    if idx > 1:
        plt.xlabel('Step')
    if idx%2 == 0:
        plt.ylabel('Loss')
    # plt.yscale('log')
    plt.xscale('log')
    plt.text(
        0.05, 0.05,
        f"\n".join(param.split('_')),
        transform=plt.gca().transAxes, 
        horizontalalignment='left', 
        verticalalignment='bottom',
        fontsize=12
    )
    plt.xlim(100, 1e6)
    # plt.ylim(1e-1, 1e1)
    plt.ylim(0, 2)
    plt.grid()
    plt.tight_layout()
    plt.legend(fontsize=12, loc='upper right', ncols=3, handlelength=1.5, frameon=False, handletextpad=0.5, columnspacing=0.5)

plt.savefig('lcurve.png', dpi=300)
