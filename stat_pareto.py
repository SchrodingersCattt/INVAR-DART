from stat_distribution import ExcelParser
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def is_dominated(target, others):
    for other in others:
        if all(o <= t for o, t in zip(other, target)) and any(o < t for o, t in zip(other, target)):
            return True
    return False

def extract_data(parser):
    tec = []
    density = []
    packing = []
    Fe, Ni, Co, Cr, V, Cu = [], [], [], [], [], []

    for row in tqdm(parser.iterate_rows()):
        tec.append(parser.get_tec(row))
        density.append(parser.get_density(row))
        Fe.append(parser.get_Fe(row))
        Ni.append(parser.get_Ni(row))
        Co.append(parser.get_Co(row))
        Cr.append(parser.get_Cr(row))
        V.append(parser.get_V(row))
        Cu.append(parser.get_Cu(row))
        packing.append((parser.get_phase(row)).replace("'", ""))

    return np.array(tec), np.array(density), np.array(Fe), np.array(Ni), np.array(Co), np.array(Cr), np.array(V), np.array(Cu), packing

def get_pareto_front(data_file, output=False):
    parser = ExcelParser(data_file)
    tec, density, Fe, Ni, Co, Cr, V, Cu, packing = extract_data(parser)

    points = np.column_stack((tec, density))

    pareto_indices = []
    for i, point in enumerate(points):
        if not is_dominated(point, np.delete(points, i, axis=0)):
            pareto_indices.append(i)
    if output:
        output_results(pareto_indices, Fe, Ni, Co, Cr, V, Cu, packing)
    return tec, density, points[pareto_indices], pareto_indices

def output_results(pareto_indices, Fe, Ni, Co, Cr, V, Cu, packing):
    pareto_elements = {
        "Fe": np.array(Fe)[pareto_indices],
        "Ni": np.array(Ni)[pareto_indices],
        "Co": np.array(Co)[pareto_indices],
        "Cr": np.array(Cr)[pareto_indices],
        "V": np.array(V)[pareto_indices],
        "Cu": np.array(Cu)[pareto_indices],
        "packing": np.array(packing)[pareto_indices]
    }

    pareto_df = pd.DataFrame(pareto_elements)
    print(pareto_df.to_markdown())
    print([list(x[:-1]) for x in pareto_df.to_numpy()])

def plot_results(tec, density, pareto_front):
    plt.figure(figsize=(8,5))
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = "Arial"
    plt.scatter(tec, density, label="Orig. data", alpha=0.5, c='#848484')
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='r', marker="s", label="Orig. Pareto front")
        
    # Extract TEC, TEC std and density from ITER_1_pred
    iter_1_pred_tec = ITER_1_pred[:, 0]
    iter_1_pred_tec_std = ITER_1_pred[:, 1]
    iter_1_pred_density = ITER_1_pred[:, 2]
    plt.scatter(iter_1_pred_tec, iter_1_pred_density, marker="<", s=60, label="Iter.1 pred.")
    plt.errorbar(iter_1_pred_tec, iter_1_pred_density, xerr=iter_1_pred_tec_std, fmt='none', elinewidth=1, capsize=3)
    
    plt.xlabel("TEC")
    plt.ylabel("Density")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    plt.tight_layout()
    plt.savefig("pareto_front.png", dpi=300)
    plt.show()


# ITER_0_pred = np.array([
#     [1.1698, 8077.52],
#     [3.2941, 7931.9691],
#     [2.4756, 8069.9608]
# ])
# ITER_0_exp = np.array([
#     [9.97, 8077.52],
#     [12.58, 7931.9691],
#     [13.67, 8069.9608]

# ])
ITER_1_pred = np.array([
    [6.07, 1.69, 8088.1],
    [5.35, 1.34, 8091.67],
    [7.04, 2.06, 8092.86]
])

if __name__ == "__main__":
    data_file = "dataset/Data_base_DFT_Thermal.xlsx"
    
    tec, density, pareto_front, pareto_indices = get_pareto_front(data_file)    
    plot_results(tec, density, pareto_front)