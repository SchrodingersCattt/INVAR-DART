import glob
import dpdata
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from deepmd.pt.infer.deep_eval import DeepProperty

MODELS = glob.glob("workspace*/model.ckpt.pt")
exclude = []
if all(item in MODELS for item in exclude):
    #MODELS.remove(exclude)
    MODELS = [item for item in MODELS if item not in exclude]



def change_type_map(origin_type: list, data_type_map, model_type_map):
    final_type = []
    for single_type in origin_type:
        element = data_type_map[single_type]
        final_type.append(np.where(np.array(model_type_map)==element)[0][0])
    return final_type

def mape(gt, pred):
    return (100/len(pred)) * np.sum(np.abs((gt-pred)/gt))

def calc_ratio(atom_numbs: list, atom_name: int):
    return atom_numbs[atom_name] / sum(atom_numbs)

val_npy_path = glob.glob("test_npy/*")

mean = 9.76257
std = 4.303086

pred_ava = []
pred_std = []
gt_ava = []

pred_all = []
gt_all = []


plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Arial'
plt.style.use('seaborn-v0_8-colorblind')
# plt.figure(figsize=(6,6))

model_list = []
for j, MODEL in enumerate(MODELS):
    model_list.append(DeepProperty(MODEL))

for i, vv in tqdm(enumerate(val_npy_path)):
    pred_ava_single = []
    gt_ava_single = []
    vs = dpdata.LabeledSystem(vv, fmt='deepmd/npy')
    
    orig_type_map = vs.data["atom_names"]
    coords = vs.data['coords']
    cells = vs.data['cells']
    for j, MODEL in enumerate(MODELS):
        
        model = model_list[j]
        atom_types = change_type_map(vs.data['atom_types'], orig_type_map, model.get_type_map())
        # atom_types = orig_type_map
        gt = vs.data["energies"]
        
        pred = model.eval(coords=coords, atom_types=atom_types, cells=cells)[0]
        pred = pred.reshape(len(pred))

        pred_ava_single.extend(pred)
        gt_ava_single.extend(gt)
    # print(pred_ava_single, gt_ava_single)
    pred_ava.append(pred_ava_single)
    gt_ava.append(gt_ava_single)


print(np.array(pred_ava).shape)
pred_std = [np.std(x) for x in np.array(pred_ava)]
pred_ava_mean = [np.mean(x) for x in np.array(pred_ava)]
gt_ava = [np.mean(x) for x in np.array(gt_ava)]
print(np.max(pred_std))

fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(3, 3, hspace=0.1, wspace=0.1)

ax_main = fig.add_subplot(gs[1:, :-1])
ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

ax_main.errorbar(gt_ava, pred_ava_mean, yerr=pred_std, fmt='none', ecolor='#884455', capsize=2, alpha=0.3)
ax_main.scatter(gt_ava, pred_ava_mean, s=15)  

print(len(gt_ava), len(gt_all))    
mae = np.mean(np.abs(np.array(gt_ava) - np.array(pred_ava_mean)))
rmse = np.sqrt(np.mean((np.array(gt_ava) - np.array(pred_ava_mean))**2))
mape_ava = mape(np.array(gt_ava), np.array(pred_ava_mean))
print("MAE: ", mae, "; RMSE: ", rmse, "; mape: ", mape_ava)
  
ax_main.text(-2, 22, rf'MAE = {mae:.3f} [$10^{{-6}}$/K]', fontsize=16)
ax_main.text(-2, 18, rf'MAPE = {mape_ava:.3f} $\%$', fontsize=16)
ax_main.plot([-4, 24], [-4, 24], 'k--')
ax_main.set_xlim([-4, 24])
ax_main.set_ylim([-4, 24])
ax_main.set_xlabel(r'TEC (Ground Truth) [$10^{-6}$/K]')
ax_main.set_ylabel(r'TEC (Prediction) [$10^{-6}$/K]')

ax_top.hist(gt_ava, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
ax_top.set_ylabel('Density')
ax_top.tick_params(labelbottom=False)

ax_right.hist(pred_ava_mean, bins=30, density=True, alpha=0.7, color='lightcoral', orientation='horizontal', edgecolor='black', linewidth=0.5)
ax_right.set_xlabel('Density')
ax_right.tick_params(labelleft=False)

plt.tight_layout()
plt.savefig('infer_by_models.png', dpi=300)




plt.figure(figsize=(15,4))
plt.subplot(131)
plt.scatter((pred_ava_mean - np.mean(pred_ava_mean)), np.abs(np.array(gt_ava) - np.array(pred_ava_mean)), s=15)
max_lim = max(max(pred_ava_mean - np.mean(pred_ava_mean)), max(np.abs(np.array(gt_ava) - np.array(pred_ava_mean))))
plt.text(0.05, 0.95, rf'Model deviation defined as:'+f'\n'+rf'$\sigma$ = prediction - <prediction>'+f'\n\n'+rf'Absolute Error defined as: '+f'\n'+rf'$\epsilon$ = prediction - ground_truth', transform=plt.gca().transAxes, ha='left', va='top', fontsize=9)
plt.xlim([0, max_lim*1.1])
plt.ylim([0, max_lim*1.1])
plt.xlabel('Model Deviation')
plt.ylabel('Absolute Error')

plt.subplot(132)
plt.scatter((pred_ava_mean-np.mean(pred_ava_mean))/np.mean(pred_ava_mean), np.abs((np.array(gt_ava) - np.array(pred_ava_mean))/np.array(gt_ava)), s=15)
max_lim = max(max((pred_ava_mean-np.mean(pred_ava_mean))/np.mean(pred_ava_mean)), max(np.abs((np.array(gt_ava) - np.array(pred_ava_mean))/np.array(gt_ava))))
plt.text(0.05, 0.95, rf'Model deviation defined as:'+f'\n'+rf'$\sigma$ = (prediction - <prediction>)/<prediction>'+f'\n\n'+rf'Relative Error defined as:'+f'\n'+rf'$\epsilon$ = (prediction - ground_truth)/ground_truth', transform=plt.gca().transAxes, ha='left', va='top', fontsize=9)
plt.xlim([0, max_lim*1.1])
plt.ylim([0, max_lim*1.1])
#plt.ylim(0,2)
#plt.xlim(0,2)
plt.xlabel('Model Deviation')
plt.ylabel('Relative Error')

plt.subplot(133)
plt.scatter(pred_std, np.sqrt((np.array(gt_ava) - np.array(pred_ava_mean))**2), s=15)
max_lim = max(max(pred_std), max(np.sqrt((np.array(gt_ava) - np.array(pred_ava_mean))**2)))
plt.text(0.05, 0.95, rf'Model deviation defined as:'+f'\n'+rf'$\sigma$ = standard deviation of prediction'+f'\n\n'+rf'RMS Error defined as: '+f'\n'+rf'$\epsilon$ = sqrt[(prediction - ground_truth)^2]', transform=plt.gca().transAxes, ha='left', va='top', fontsize=9)
plt.xlim([0, max_lim*1.1])
plt.ylim([0, max_lim*1.1])
plt.xlabel('Model Deviation')
plt.ylabel('RMS Error')

plt.tight_layout()
plt.savefig('infer_by_models_error.png', dpi=300)


