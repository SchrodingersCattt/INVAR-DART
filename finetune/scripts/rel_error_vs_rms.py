import glob
import dpdata
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from deepmd.pt.infer.deep_eval import DeepProperty

ITER = "iter01"
MODEL_TYPE = "dpa_v2_b4" if ITER == "iter00" else "dpa_v3_1"
val_npy_path = glob.glob("new_data_npy/*")

MODELS = glob.glob(f"../../{ITER}.finetune/{MODEL_TYPE}/workspace*/model.ckpt.pt")
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

# Create the main visualization: std vs rms error
plt.figure(figsize=(8, 6))
rms_errors = np.sqrt((np.array(gt_ava) - np.array(pred_ava_mean))**2)
plt.scatter(pred_std, rms_errors, alpha=0.7, s=30)
plt.xlabel('Standard Deviation of Predictions')
plt.ylabel('RMS Error')
plt.title('Standard Deviation vs RMS Error')

upper = max(np.max(pred_std), np.max(rms_errors))
plt.xlim([0, upper*1.1])
plt.ylim([0, upper*1.1])


plt.tight_layout()
plt.savefig(f'std_vs_rms_{ITER}_{MODEL_TYPE}.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some statistics
print(f"Max std: {np.max(pred_std):.4f}")
print(f"Min std: {np.min(pred_std):.4f}")
print(f"Mean std: {np.mean(pred_std):.4f}")
print(f"Max RMS error: {np.max(rms_errors):.4f}")
print(f"Min RMS error: {np.min(rms_errors):.4f}")
print(f"Mean RMS error: {np.mean(rms_errors):.4f}")