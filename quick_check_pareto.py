from stat_distribution import ExcelParser
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

STD_FACTOR = 2
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
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = "Arial"
    plt.scatter(tec, density, label="Orig. data", alpha=0.5, c='#848484')
    
    # Identify outliers: TEC<3 and density<8200
    outlier_mask = (pareto_front[:, 0] < 3) & (pareto_front[:, 1] < 8200)
    outliers = pareto_front[outlier_mask]
    non_outliers = pareto_front[~outlier_mask]
    
    # Sort non-outliers for plotting the Pareto front line
    sorted_pareto_front = non_outliers[non_outliers[:, 0].argsort()]
    plt.plot(sorted_pareto_front[:, 0], sorted_pareto_front[:, 1], 'r--s', label="Orig. Pareto front")
    
    # Plot outliers as hollow circles
    if len(outliers) > 0:
        plt.scatter(outliers[:, 0], outliers[:, 1], facecolors='none', edgecolors='red', s=50, label="Suspected outliers", linewidths=1.5)

    # Extract TEC, TEC std and density from monitor_pred
    monitor_pred_tec = monitor_pred[:, 0]
    monitor_pred_tec_std = monitor_pred[:, 1]
    monitor_pred_density = monitor_pred[:, 2]
    plt.scatter(monitor_pred_tec, monitor_pred_density, marker="s", s=100, c='#3da4ff')
    plt.errorbar(monitor_pred_tec, monitor_pred_density, xerr=monitor_pred_tec_std * STD_FACTOR, fmt='none', elinewidth=2, capsize=4, c='#3da4ff')
    plt.text(0.05, 0.95, f"The error bars represent {STD_FACTOR}×std", transform=plt.gca().transAxes)
    plt.xlabel("TEC")
    plt.ylabel("Density")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    plt.tight_layout()

    # ax_inset = plt.axes([0.7, 0.35, 0.25, 0.4])  # [left, bottom, width, height] 相对坐标
    # ax_inset.scatter(tec, density, alpha=0.5, c='#848484')
    # ax_inset.plot(sorted_pareto_front[:, 0], sorted_pareto_front[:, 1], 'r--s')

    # if len(outliers) > 0:
    #     ax_inset.scatter(outliers[:, 0], outliers[:, 1], facecolors='none', edgecolors='red', s=50, linewidths=1.5)
    # ax_inset.scatter(monitor_pred_tec, monitor_pred_density, marker="<", s=60, c='#3da4ff')
    # ax_inset.errorbar(monitor_pred_tec, monitor_pred_density, 
    #                   xerr=monitor_pred_tec_std, fmt='none', 
    #                   elinewidth=1, capsize=3, c='#3da4ff')
    # ax_inset.set_xlim(-2, 5)
    # ax_inset.set_ylim(8000, 8400)

    plt.savefig("pareto_front_monitor.png", dpi=300)
    plt.show()


monitor_pred = np.array([
    [16.2588, 1.5955, 6577.64],
    [9.3925, 1.848, 8127.51]
])

if __name__ == "__main__":
    data_file = "dataset/Data_base_DFT_Thermal.xlsx"
    
    tec, density, pareto_front, pareto_indices = get_pareto_front(data_file)    
    plot_results(tec, density, pareto_front)